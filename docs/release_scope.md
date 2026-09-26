# Public research materials

| Requested material | This release |
|---|---|
| Inspectable ML implementation | Model, training and inference source |
| Pretrained models | Primary 20260416 checkpoint plus five identified ablations |
| Raw HMI/SP access | Archive references and parameterised download commands |
| Alignment, cropping, dataset generation | Coarse affine pipeline and local registration/NPZ pipeline |
| Training data access | Exact split and three examples; full archive explicitly deferred |
| Derived full-disk product access | Explicitly deferred to a separate archive |
| Evaluation definitions | Source and mathematical conventions, including numerical caveats |
| AR/non-AR and angle analysis | Shared evaluator and all test metadata |
| Reproducible examples | Input, reference, existing prediction and existing figures |
| Off-limb fallback flags and uncertainty | Required for a future full-disk archive; no unsupported claim of existing masks or uncertainty estimates |

The primary publication request and the data-transparency request extend beyond code. Training and derived data remain outstanding until their separate archive is available. Implementation conventions and the supplied historical training configurations are documented separately.

Only a single shared statistics pipeline is needed. Including the grouping logic and sample IDs is important because population selection and aggregation conventions affect the tables. Repeated per-model plotting scripts do not add reproducibility.
