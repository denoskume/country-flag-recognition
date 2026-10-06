# Semantic Conversation Understanding — Flag Entry Addendum

This addendum extends `2026-10-06-semantic-conversation-understanding-design.md` with the user-approved flag-upload entry rule demonstrated by the Senegal regression.

## Flag Upload Entry Rule

Uploading a flag establishes country identity only. It must never establish report scope, mark a semantic brief ready, or trigger report generation by itself.

After successful recognition, the application must enter `awaiting_interest` with an empty report brief and ask one short open question about what the user wants to learn about the recognized country. The flag-recognition event is system context, not a synthetic user report request and must not be passed through report-readiness inference as though the user had requested content.

A text message submitted in the same turn as a flag upload is different: the uploaded flag supplies country identity and the accompanying user text is the first genuine report-requirement turn. That text may immediately produce clarification or generation when semantically sufficient.

## Acceptance Criteria

1. Uploading a Senegal flag with no text leaves the report brief empty, does not call report generation, and asks what the user wants to learn about Senegal.
2. Uploading any flag with no text can never produce `Generating your report now.`.
3. Uploading a flag together with a clear request such as `traditional music after independence` uses the flag as country context and the text as the semantic brief.
4. A new flag upload clears stale report scope from the previous country before asking for the new country's need.
5. Existing uploaded-flag rendering, 2 cm × 1 cm preview sizing, recognition, and later report generation remain intact.
