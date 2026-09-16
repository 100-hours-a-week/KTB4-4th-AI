from app.core.config import Settings


def test_v2_model_runtime_is_a_configuration_change() -> None:
    v1 = Settings(
        app_env="test",
        response_model_base_url="https://provider.example.com/v1",
        extraction_model_base_url="https://provider.example.com/v1",
        response_model_name="provider-response",
        extraction_model_name="provider-extraction",
    )
    v2 = Settings(
        app_env="test",
        response_model_base_url="https://gpu.example.com/response/v1",
        extraction_model_base_url="https://gpu.example.com/extract/v1",
        response_model_name="kanana-response",
        extraction_model_name="kanana-extractor",
    )

    assert str(v1.response_model_base_url) != str(v2.response_model_base_url)
    assert v2.response_model_name == "kanana-response"
    assert v2.extraction_model_name == "kanana-extractor"
