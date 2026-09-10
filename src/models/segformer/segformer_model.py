from transformers import SegformerConfig, SegformerForSemanticSegmentation

def build_segformer_model_binary(
    pretrained_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels: int = 1,
):
    config = SegformerConfig.from_pretrained(pretrained_model_name)

    config.num_labels = num_labels
    config.id2label = {0: "lesion"}
    config.label2id = {"lesion": 0}
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    return model

def build_segformer_model_multiclass(
    pretrained_model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels: int = 1,
):
    config = SegformerConfig.from_pretrained(pretrained_model_name)

    config.num_labels = num_labels
    config.problem_type = None
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    return model