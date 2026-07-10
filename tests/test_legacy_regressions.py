from tools.test_relation_vocative_gate import (
    test_block_true_vocative_offscene,
    test_no_reassign_nonvocative_offscene,
    test_reassign_true_vocative_in_scene,
    test_vocative_truth_table,
)
from tools.test_script_block_split import (
    test_apply_splits_narration_only,
    test_detection,
    test_no_false_split,
    test_split_and_alias,
)


def test_relation_vocative_gate_regression():
    for test in (test_vocative_truth_table, test_no_reassign_nonvocative_offscene,
                 test_reassign_true_vocative_in_scene, test_block_true_vocative_offscene):
        test()


def test_script_block_split_regression():
    for test in (test_detection, test_split_and_alias, test_apply_splits_narration_only, test_no_false_split):
        test()
