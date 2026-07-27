import unittest
from types import SimpleNamespace

from sglang.srt.layers.dp_attention import _is_glm_dsa_nvfp4_moe_model


class TestGlmDsaIdlePadding(unittest.TestCase):
    def _model_config(
        self,
        *,
        arch="GlmMoeDsaForCausalLM",
        quantization=None,
        quantization_config=None,
        nvfp4_moe_meta=None,
    ):
        return SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[arch],
                quantization_config=quantization_config,
            ),
            quantization=quantization,
            nvfp4_moe_meta=nvfp4_moe_meta,
        )

    def test_detects_glm_dsa_modelopt_fp4(self):
        self.assertTrue(
            _is_glm_dsa_nvfp4_moe_model(
                self._model_config(quantization="modelopt_fp4")
            )
        )

    def test_detects_glm_dsa_hf_nvfp4_quant_config(self):
        self.assertTrue(
            _is_glm_dsa_nvfp4_moe_model(
                self._model_config(quantization_config={"quant_algo": "NVFP4"})
            )
        )

    def test_ignores_non_glm_dsa_nvfp4(self):
        self.assertFalse(
            _is_glm_dsa_nvfp4_moe_model(
                self._model_config(
                    arch="DeepseekV3ForCausalLM",
                    quantization="modelopt_fp4",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
