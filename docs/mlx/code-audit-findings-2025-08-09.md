# Thea Code – Architecture & Test Audit (2025-08-09)

This document consolidates code inspection findings, architectural conformance notes, and a concrete test hardening plan before dependency cleanup.

## Summary
- Architecture is coherently provider-agnostic via neutral history/types and a unified MCP tool-use path.
- MCP routing and transports are modular, with test-aware guards.
- Most providers conform to neutral types and delegate tool use to MCP; Anthropic/Vertex use neutral clients.
- Remaining work centers on edge-case tests, lifecycle robustness, coverage, and safe SDK cleanup.

## Key Findings by Area

### Neutral contracts
- File: `src/shared/neutral-history.ts`
- Defines text/image/tool_use/tool_result blocks and links via IDs. Providers adhere to these types in message conversion and tool routing.

### MCP conversion and routing
- Files: `src/services/mcp/core/McpConverters.ts`, `src/utils/json-xml-bridge.ts`, `src/services/mcp/core/McpToolRouter.ts`, `src/services/mcp/core/McpToolRegistry.ts`
- xml/json/openai → neutral conversion: validations guard required fields; errors surface with messages.
- `mcpToXml` escapes XML attributes and supports text/images (base64/url), tool_use, and nested tool_result, logging unknown types.
- Router detects formats (XML/JSON/OpenAI/neutral) and returns results in original format, including error cases.
- Registry is evented and used by executor/provider to reflect tool lifecycle.

### MCP execution and transports
- Files: `src/services/mcp/core/McpToolExecutor.ts`, `src/services/mcp/providers/EmbeddedMcpProvider.ts`, `src/services/mcp/transport/SseTransport.ts`, `src/services/mcp/transport/StdioTransport.ts`
- Executor: singleton; queues pending registrations; forwards started/stopped/tool-{un}registered; exposes server URL.
- Embedded provider: SSE vs stdio; registers tools/resources/templates; connects SDK transport; robust port discovery and test guards.
- SSE transport: dynamic import of SDK; Jest teardown short-circuit; `getPort` and `getUnderlyingTransport` for server.connect.
- Stdio transport: SDK import with fallback mock; stderr exposure; simple lifecycle.

### Provider handlers
- Anthropic (`src/api/providers/anthropic.ts`): uses `NeutralAnthropicClient` (fetch-based). Thinking budget clamped; tool_use routed via BaseProvider; token counting via client.
- OpenAI (`src/api/providers/openai.ts`): neutral conversion; stream parsing with XmlMatcher; tool_calls detection via shared utils; MCP tool routing; usage chunks.
- Ollama (`src/api/providers/ollama.ts`): delegates tool-call detection to OpenAI handler; HybridMatcher for reasoning; MCP routing.
- Vertex (`src/api/providers/vertex.ts`): uses `NeutralVertexClient` for Claude/Gemini; neutral history; thinking variants normalized; MCP routing.
- Bedrock (`src/api/providers/bedrock.ts`): AWS SDK (expected); ARN validation; cross-region inference; neutral conversions.
- Glama (`src/api/providers/glama.ts`): OpenAI-compatible base; neutral path; completePrompt aligns.

### Hub and settings
- File: `src/services/mcp/management/McpHub.ts`
- JSON/JSONC parsing; shape normalization (servers→mcpServers); mixed-field validation with improved error messaging; test-mode guards for watchers.

### Test infra
- Files: `jest.config.js`, `jest.sse.config.js`, `src/__mocks__/jest.setup.ts`, `test/global{Setup,Teardown}.ts`
- Global setup/teardown set flags (`__JEST_TEARDOWN__`) and clean mock servers; logs suppressed post-teardown; SSE import edge cases addressed.

## Gaps / Risks
- Coverage gaps unknown until baseline is run; likely low in conversion edge cases, router error paths, transport guards.
- SSE import path and port detection have complexity—worth stress tests for lifecycle and retries.
- Bedrock handler is complex (ARN/region/cross-region); needs negative tests.
- Package lists `@anthropic-ai/sdk`; code has no live imports—safe removal pending green builds.

## Targeted Test Plan

1) Converters and format detection
- McpConverters.xmlToMcp: valid tool_use, missing fields, nested, bad JSON in input.
- McpConverters.jsonToMcp: string/object inputs; required props enforcement.
- McpConverters.openAiToMcp: function_call and tool_calls shapes; invalid input errors.
- McpConverters.mcpToXml: text/images (base64/url), tool_use with embedded quotes/newlines, nested tool_result, error payload; unknown types.
- JsonMatcher/FormatDetector: nested braces, partial chunks, buffer cap spill, thinking/tool_use objects; xml vs json detection.

2) Router and executor
- McpToolRouter.detectFormat: XML/JSON/OpenAI/neutral object; ambiguous defaults.
- Round-trip (XML/JSON/OpenAI→neutral→execute→neutral→original): preserves format and tool_use_id; error results.
- McpToolExecutor lifecycle: init/shutdown multiple times; pendingRegistrations flush; events forwarded; getServerUrl; error envelopes.

3) Transports and provider
- SseTransport: successful dynamic import mocked; teardown short-circuit; getPort; underlying transport presence.
- StdioTransport: fallback mock when SDK missing; stderr/handlers.
- EmbeddedMcpProvider (SSE): port=0 assignment; restart gets a new port; started/stopped events; serverUrl set.

4) Providers
- Anthropic: thinking budget clamping; tool_use round-trip; countTokens happy-path and error.
- OpenAI: streaming branch, XmlMatcher for reasoning, tool_calls → MCP routing, non-streaming branch.
- Ollama: delegation to OpenAI handler; HybridMatcher reasoning; tool_result chunk shape.
- Vertex: Claude and Gemini paths; thinking variants normalized; completePrompt helpers.
- Bedrock: validateBedrockArn variants; cross-region prefixing; creds modes; yield errors and usage.
- Glama: OpenAI-compatible path; supportsTemperature gate; non-streaming completion.

5) Misc
- BaseProvider.registerTools schemas registered only; execution via MCP provider.
- attemptCompletionTool: partial vs final; approvals; telemetry/events; Neutral blocks filtering; feedback assembly.
- TheaTaskHistory: NeutralConversationHistory file IO; delete cleanup order; export path.
- port-utils: findAvailablePort/waitForPortInUse retries/timeouts and test-mode fast paths.
- Logging/i18n test mode: noop logger in Jest; console suppression; i18n avoids external loads.

## Coverage & CI
- Run Mocha + Jest with coverage; store per-file HTML reports.
- Set thresholds: 80% core MCP/Providers, 70% overall; enforce in CI.
- Track flaky tests from existing summaries; add retries/guards where appropriate.

## Dependency Cleanup (after green)
- Remove `@anthropic-ai/sdk` from package.json once builds/tests pass; update docs and changelog explaining neutral client usage.
- Keep AWS SDK for Bedrock as intentional.

## Actionable Next Steps
- Generate coverage baseline; identify <70% files to prioritize.
- Implement converter + router round-trip tests first (highest leverage for MCP path).
- Add SSE lifecycle tests (dynamic ports, retries, teardown guard).
- Tackle Bedrock and Vertex provider edge cases.
- Prepare PR to remove `@anthropic-ai/sdk` gated on green CI.
