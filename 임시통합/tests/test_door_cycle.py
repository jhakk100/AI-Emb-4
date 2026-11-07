import unittest

from door_cycle import DoorCycle


class DoorCycleTests(unittest.TestCase):
    def test_continuous_opens_closes_then_returns_to_stage_one(self):
        cycle = DoorCycle()
        self.assertTrue(cycle.start(0))
        self.assertEqual(cycle.phase, "open")
        self.assertIsNone(cycle.advance(2.99))
        self.assertEqual(cycle.advance(3), "closed")
        self.assertEqual(cycle.phase, "closed")
        self.assertIsNone(cycle.advance(4.49))
        self.assertEqual(cycle.advance(4.5), "reset")
        self.assertEqual(cycle.phase, "waiting")
        self.assertTrue(cycle.start(5))

    def test_once_completes_after_door_closes_and_requires_manual_restart(self):
        cycle = DoorCycle(mode="once")
        cycle.start(10)
        self.assertEqual(cycle.advance(13), "closed")
        self.assertFalse(cycle.completed)
        self.assertEqual(cycle.advance(14.5), "finished")
        self.assertTrue(cycle.completed)
        self.assertFalse(cycle.start(20))
        cycle.reset()
        self.assertTrue(cycle.start(21))

    def test_repeated_success_does_not_extend_open_time_or_reopen_while_closed(self):
        cycle = DoorCycle()
        cycle.start(0)
        self.assertFalse(cycle.start(2))
        self.assertEqual(cycle.advance(3), "closed")
        self.assertFalse(cycle.start(4))
        self.assertEqual(cycle.advance(4.5), "reset")

    def test_mode_change_during_open_is_applied_after_closing(self):
        cycle = DoorCycle()
        cycle.start(0)
        cycle.mode = "once"
        self.assertEqual(cycle.advance(3), "closed")
        self.assertEqual(cycle.advance(4.5), "finished")


if __name__ == "__main__":
    unittest.main()
