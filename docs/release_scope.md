# Public research materials

| Requested material | This release |
|---|---|
| Inspectable ML implementation | Model, training and inference source |
| Pretrained models | Primary PM-LTEW-CC checkpoint plus five identified ablations |
| Raw HMI/SP access | Archive references and parameterised download commands |
| Alignment, cropping, dataset generation | Coarse affine pipeline and local registration/NPZ pipeline |
| Training data access | Exact split and three examples; full archive explicitly deferred |
| Derived full-disk product access | Three paper figures and case metadata; complete generation code; large arrays deferred |
| Evaluation definitions | Source and mathematical conventions, including numerical caveats |
| AR/non-AR and angle analysis | Shared evaluator and all test metadata |
| Reproducible examples | Input, reference, existing prediction and existing figures |
| Off-limb fallback flags and uncertainty | New generation code emits model/interpolation/invalid masks; no historical masks or uncertainty estimates claimed |

The primary publication request and the data-transparency request extend beyond code. The complete paired training dataset remains scheduled for a separate archive; three full-disk previews and their generation workflow are included here. Implementation conventions and the supplied historical training configurations are documented separately.

Only a single shared statistics pipeline is needed. Including the grouping logic and sample IDs is important because population selection and aggregation conventions affect the tables. Repeated per-model plotting scripts do not add reproducibility.
