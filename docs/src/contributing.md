# Contributing to the docs

This book is an [mdBook](https://rust-lang.github.io/mdBook/). The sources live
under `docs/src/`, indexed by `docs/src/SUMMARY.md`.

## Building

```bash
mdbook serve docs --open   # live-reload preview at http://localhost:3000
mdbook build docs          # render to docs/book/ (gitignored)
```

The embedded example programs are compiled by `cargo build --examples` in the
workspace, which is the authoritative check that the code in this book builds.

## Conventions

Keep the book to these rules so it stays correct as the crates evolve:

1. **The API reference lives in rustdoc, not here.** Do not restate signatures or
   duplicate `///` documentation. Link into the docs.rs pages instead, using
   reference-style links collected at the bottom of the page, for example:

   ```markdown
   See [`Device::upload`] for the contract.

   [`Device::upload`]: https://docs.rs/et-rs/latest/et_soc1/struct.Device.html#method.upload
   ```

2. **Runnable examples come from real files.** A full program should be embedded
   from `et-rs/examples/` with an include, so the book cannot drift from what
   actually builds. The example depends on the crate (and the SDK at run time), so
   it is not standalone-compilable by mdBook; mark it `rust,ignore` and rely on
   `cargo build --examples` as the compile check. Include paths are resolved
   relative to the including file's directory:

   ````markdown
   ```rust,ignore
   \{{#include ../../../et-rs/examples/reduce.rs}}
   ```
   ````

   Prefer this over pasting code. Illustrative fragments are also `rust,ignore`.

3. **One concept per page**, listed in `SUMMARY.md`. User-guide pages are
   task-oriented (how to do X); developer-guide pages explain how and why the
   internals work.

4. **House style** (matching the crates): British English spelling, no em dashes,
   plain ASCII punctuation, and prose pitched at a reader who knows the domain.

## Structure

- `user-guide/` : installing, first program, device memory, launching kernels,
  the demos. What a consumer of the crates needs.
- `developer-guide/` : architecture, wire protocol, writing kernels, the
  coherence model, regenerating bindings. What a contributor needs.

Add a new page by creating the `.md` file and linking it from `SUMMARY.md`.
