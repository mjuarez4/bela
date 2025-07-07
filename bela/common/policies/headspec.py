from bela.typ import Head, HeadSpec, Morph

def build_headspec(state_features: dict) -> HeadSpec:
    def compute_head(items, prefix):
        total = sum(v.shape[0] for k, v in items if prefix in k)
        return (total,)

    return HeadSpec(
        robot=Head(Morph.ROBOT, compute_head(state_features.items(), "robot")),
        human=Head(Morph.HUMAN, compute_head(state_features.items(), "human")),
        share=Head(Morph.HR, compute_head(state_features.items(), "shared")),
    )

