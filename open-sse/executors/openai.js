import { DefaultExecutor } from "./default.js";
import { HTTP_STATUS } from "../config/runtimeConfig.js";
import { openaiToOpenAIResponsesRequest } from "../translator/request/openai-responses.js";
import { openaiResponsesToOpenAIResponse } from "../translator/response/openai-responses.js";
import { initState } from "../translator/index.js";
import { parseSSELine, formatSSE } from "../utils/streamHelpers.js";
import { proxyAwareFetch } from "../utils/proxyFetch.js";
import { SSE_DONE } from "../utils/sseConstants.js";

export class OpenAIExecutor extends DefaultExecutor {
  constructor() {
    super("openai");
    this.knownResponsesModels = new Set();
  }

  requiresResponsesEndpoint(model) {
    return /gpt-5|o[134]-/i.test(model);
  }

  buildUrl(model, stream, urlIndex = 0, credentials = null) {
    const rt = credentials?.runtimeTransport;
    if (rt?.baseUrl) {
      return rt.urlSuffix ? `${rt.baseUrl}${rt.urlSuffix}` : rt.baseUrl;
    }

    return this.config.baseUrl;
  }

  async execute(options) {
    const { model, log } = options;

    if (this.knownResponsesModels.has(model)) {
      log?.debug("OPENAI", `Using cached /responses route for ${model}`);
      return this.executeWithResponsesEndpoint(options);
    }

    const result = await super.execute(options);

    if (result.response.status === HTTP_STATUS.BAD_REQUEST && this.requiresResponsesEndpoint(model)) {
      const errorBody = await result.response.clone().text();

      if (errorBody.includes("reasoning_effort") || errorBody.includes("/v1/responses") || errorBody.includes("'messages'") || errorBody.includes("'input'")) {
        log?.warn("OPENAI", `Model ${model} requires /responses. Switching...`);
        this.knownResponsesModels.add(model);
        return this.executeWithResponsesEndpoint(options);
      }
    }

    return result;
  }

  async executeWithResponsesEndpoint({ model, body, stream, credentials, signal, log, proxyOptions = null }) {
    const url = this.config.responsesUrl;
    const headers = this.buildHeaders(credentials, stream);

    const transformedBody = openaiToOpenAIResponsesRequest(model, body, stream, credentials);

    log?.debug("OPENAI", "Sending translated request to /responses");

    const response = await proxyAwareFetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(transformedBody),
      signal
    }, proxyOptions);

    if (!response.ok) {
      return { response, url, headers, transformedBody };
    }

    const state = initState("openai-responses");
    state.model = model;

    const decoder = new TextDecoder();
    let buffer = "";

    const transformStream = new TransformStream({
      async transform(chunk, controller) {
        buffer += decoder.decode(chunk, { stream: true });
        const lines = buffer.split("\n");

        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          const parsed = parseSSELine(trimmed);
          if (!parsed) continue;

          if (parsed.done && stream === true) {
            controller.enqueue(new TextEncoder().encode(SSE_DONE));
            continue;
          }

          const converted = openaiResponsesToOpenAIResponse(parsed, state);
          if (converted) {
            const sseString = formatSSE(converted, "openai");
            controller.enqueue(new TextEncoder().encode(sseString));
          }
        }
      },
      flush(controller) {
        if (buffer.trim()) {
          const parsed = parseSSELine(buffer.trim());
          if (parsed && !parsed.done) {
            const converted = openaiResponsesToOpenAIResponse(parsed, state);
            if (converted) {
              controller.enqueue(new TextEncoder().encode(formatSSE(converted, "openai")));
            }
          }
        }
      }
    });

    if (!response.body) {
      return { response: new Response("", { status: response.status, headers: response.headers }), url, headers, transformedBody };
    }
    const convertedStream = response.body.pipeThrough(transformStream);

    return {
      response: new Response(convertedStream, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers
      }),
      url,
      headers,
      transformedBody
    };
  }
}

export default OpenAIExecutor;
