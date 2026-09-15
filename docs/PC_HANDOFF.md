# Talos release handoff

The user stopped capability development and approved finishing the submission. The final requested work is repository/privacy cleanup, current media uploads, review of the descriptions, and public GitHub/Hugging Face access. The user explicitly authorized both services to become public on September 14. **Never click the final submission button.** The user performs that action.

Read [README](../README.md), [setup](SETUP.md), [evidence and limits](EVIDENCE.md), [archive](ARCHIVE.md) and [submission files](../submission/README.md). These current documents supersede dated instructions in the development archive. Do not resume the broad robotics experiments automatically.

The delivered video is 289.5 seconds, 1920×1080, H.264/AAC. It uses the user's narration and recordings; raw sources and editing intermediates must stay local. Its SHA-256 is `6705fe453e046745a3d9c08355c1051d572569b51dbfabcaa5393aa8ef022e43`. Only the reviewed final MP4 is intended for the submission video field. The current eight-slide presentation is `submission/presentation.pdf`, SHA-256 `39fd47e3568bababa04a7157a8a6905c7391113f411ef095bc9408a6fe065a6f`. It uses Delphi's visual style without copying its slide layouts.

The demonstrated six-skill controller, limited live mug correction, voice and bottle relays work within the measured scope. General jointly randomized seven-item table setting is unfinished. Core Ultra Series 2/3 eligibility remains unresolved; legacy Intel evidence must not be described as qualifying hardware. Current result counts and denominators are in the evidence index.

Preserve published model bytes and both successful and failed evidence. The annotated development snapshot and private local archive retain removed material. Do not edit frozen runtime dependencies to simplify the file tree: the promoted profile verifies their hashes.

Persistent preferences:

- Never commit credentials, private recordings, personal paths or personal author/committer metadata. Use `Talos contributors <contributors@talos.invalid>` and explicit UTC dates. Do not push old laptop backups or branches.
- Keep `.env` local and ignored. Ask the user only when a microphone or missing key is actually required; existing Speechmatics and hosting configuration is already local.
- Before every dataset, capture or export, check the actual destination filesystem for 10 GiB free **plus expected writes**. Preserve existing outputs; choose unused directories and console filenames.
- Keep the user's port-8770 demo running. Do not reset or restart it during release preparation.
- User authorization persists. Public release is now authorized; final Submit remains reserved for the user.
