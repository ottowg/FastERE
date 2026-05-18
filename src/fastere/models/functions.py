def getBertForMaskedLMClass(model_config):
    if model_config.model_type == "roberta":
        from transformers import RobertaForMaskedLM

        return RobertaForMaskedLM
    elif model_config.model_type == "bert":
        from transformers import BertModel

        return BertModel
    elif model_config.model_type == "albert":
        from transformers import AlbertForMaskedLM

        return AlbertForMaskedLM
    raise ValueError(f"Unsupported model_type: {model_config.model_type}")


def getPretrainedLMHead(model, model_config):
    if model_config.model_type == "roberta":
        return model.lm_head
    elif model_config.model_type == "bert":
        return model.cls.predictions
    elif model_config.model_type == "albert":
        return model.predictions
    raise ValueError(f"Unsupported model_type: {model_config.model_type}")
