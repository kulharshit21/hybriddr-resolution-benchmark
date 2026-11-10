# Public-release boundaries

The core research source was copied into a new repository, without importing the private project history. Only Python source, experiment-configuration JSON, dependency text and code documentation are included. Model/loss code and numerical hyperparameters are preserved. No retinal image, mask, checkpoint, weight, prediction CSV, cache array, experimental result or private review log is included.

Local MAMAGuard termination/release tools, migration helpers and cloud transfer/export scripts are omitted. Unit tests dedicated solely to omitted process-stop helpers are also omitted; they are not converted to passing no-ops. The V3 legacy process-stop flag now fails with an explicit message. Hard-coded campus project-directory defaults in dashboards use the caller's current directory. These are release portability changes, not training-method changes.

The original dependency pins and later recorded package inventory are separate historical records. They are not represented as a newly solved exact lockfile. Model providers and dataset links are in README. Public pairing and acquisition limitations remain documented. No upstream model implementation, model-weight blob or dataset archive is vendored.
