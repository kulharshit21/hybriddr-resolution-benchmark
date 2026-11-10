import unittest

from publication_model_preflight import EXPECTED_CNN, EXPECTED_VIT, validate_config


class PublicationModelPreflightTests(unittest.TestCase):
    def valid(self):
        return {
            'tiny': False,
            'vit': EXPECTED_VIT,
            'cnn': EXPECTED_CNN,
            'pretrained': True,
            'image_size': 224,
        }

    def test_accepts_exact_publication_encoder_contract(self):
        self.assertIs(validate_config(self.valid())['pretrained'], True)

    def test_rejects_tiny_or_changed_encoder_contract(self):
        for key, value in (
            ('tiny', True), ('vit', 'vit_base_patch14_dinov2.lvd142m'),
            ('cnn', 'convnext_small.fb_in22k_ft_in1k'),
            ('pretrained', False), ('image_size', 56),
        ):
            with self.subTest(key=key):
                config = self.valid()
                config[key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)


if __name__ == '__main__':
    unittest.main(verbosity=2)
