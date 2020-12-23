import unittest
from unittest.mock import Mock
from unittest.mock import patch

from python_app.sp_digitiser import SpDigitiser


class TestSpDigitiser(unittest.TestCase):
    def test_missing_param(self):
        with self.assertRaises(RuntimeError):
            sut = SpDigitiser({"number_of_records": 1})

    def test__invalid_number_of_samples(self):
        with self.assertRaises(RuntimeError):
            sut = SpDigitiser(
                {
                    "number_of_records": 1000,
                    "samples_per_record": 1000000,
                    "delay": 0,
                    "trigger_type": SpDigitiser.TRIGGER_EXTERNAL,
                    "channelA_gain": 1,
                    "channelB_gain": 1,
                    "channelA_offset": 0,
                    "channelB_offset": 0,
                }
            )

    def test__invalid_number_of_records(self):
        with self.assertRaises(RuntimeError):
            sut = SpDigitiser(
                {
                    "number_of_records": 1000000,
                    "samples_per_record": 10,
                    "delay": 0,
                    "trigger_type": SpDigitiser.TRIGGER_EXTERNAL,
                    "channelA_gain": 1,
                    "channelB_gain": 1,
                    "channelA_offset": 0,
                    "channelB_offset": 0,
                },
            )

    def test__ok(self):
        sut = SpDigitiser(
            {
                "number_of_records": 10,
                "samples_per_record": 10,
                "delay": 0,
                "trigger_type": SpDigitiser.TRIGGER_EXTERNAL,
                "channelA_gain": 1,
                "channelB_gain": 1,
                "channelA_offset": 0,
                "channelB_offset": 0,
            }
        )
        assert True
