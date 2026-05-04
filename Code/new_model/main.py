import argparse

from .causal import CausalConfig, compute_tpr_at_target_fpr, load_causal_for_mnw, train_causal
from .config import (
    DATA_DIR,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_MAX_AI_SAMPLES,
    DEFAULT_NUM_REAL_TEST_SAMPLES,
    DEFAULT_REPORTS_SAVE_PATH,
    DEFAULT_ROC_SAVE_PATH,
    DEFAULT_ROC_TARGET_FPR,
    DEFAULT_TEST_RESULTS_PATH,
    DEFAULT_THRESHOLD,
    DEFAULT_TRAIN_BATCH_SIZE,
    MNW_AI_IMAGES_DIR,
    OUTPUT_DIR,
)
from .utils import (
    collect_mnw_evaluation_outputs,
    get_hf_token,
    plot_roc_with_target_fpr,
    resolve_device,
    resolve_output_path,
    save_evaluation_reports,
    save_test_results,
    summarize_mnw_evaluation,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate the causal FIND model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-causal", help="Train the causal FIND model.")
    add_common_train_args(train_parser)
    train_parser.add_argument("--causal-mode", choices=["cls_only", "partial", "full"], default="full")
    train_parser.add_argument("--ood-generator-fraction", type=float, default=0.25)
    train_parser.add_argument("--ood-seed", type=int, default=42)

    test_parser = subparsers.add_parser("test-causal", help="Run MNW evaluation for the causal FIND model.")
    add_common_test_args(test_parser)
    return parser


def add_common_train_args(parser):
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--max-ai-samples", type=int, default=DEFAULT_MAX_AI_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--noise-std", type=float)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--token", default=get_hf_token())


def add_common_test_args(parser):
    parser.add_argument("--val-dir", default=MNW_AI_IMAGES_DIR)
    parser.add_argument("--num-real-test-samples", type=int, default=DEFAULT_NUM_REAL_TEST_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--threshold-from-target-fpr",
        type=float,
        help="Select the decision threshold from ROC so the realized FPR is at or below this target.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--token", default=get_hf_token())
    parser.add_argument("--roc-target-fpr", type=float, default=DEFAULT_ROC_TARGET_FPR)
    parser.add_argument("--roc-save-path", default=DEFAULT_ROC_SAVE_PATH)
    parser.add_argument("--results-save-path", default=DEFAULT_TEST_RESULTS_PATH)
    parser.add_argument("--reports-save-path", default=DEFAULT_REPORTS_SAVE_PATH)


def apply_causal_overrides(config: CausalConfig, args):
    config.data_dir = args.data_dir
    config.max_ai_samples = args.max_ai_samples
    config.batch_size = args.batch_size
    config.output_dir = args.output_dir
    config.device = resolve_device(args.device)
    config.token = args.token
    config.causal_mode = args.causal_mode
    config.ood_generator_fraction = args.ood_generator_fraction
    config.ood_seed = args.ood_seed
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.lr = args.lr
    if args.noise_std is not None:
        config.noise_std = args.noise_std
    return config


def run_test(model, config, args):
    raw_result = collect_mnw_evaluation_outputs(
        model,
        device=config.device,
        token=getattr(config, "token", args.token),
        val_dir=args.val_dir,
        num_real_test_samples=args.num_real_test_samples,
        batch_size=args.batch_size,
    )

    result = summarize_mnw_evaluation(
        raw_result,
        threshold=args.threshold,
        report_title=f"Joint Evaluation Final Report | Fixed Threshold = {args.threshold:.4f}",
    )
    report_payload = {
        "fixed_threshold": build_report_record(result, mode="fixed_threshold"),
        "target_fpr_reports": {},
    }

    target_fprs = [0.05, 0.01]
    if args.threshold_from_target_fpr is not None and args.threshold_from_target_fpr not in target_fprs:
        target_fprs.append(args.threshold_from_target_fpr)

    for target_fpr in target_fprs:
        target_tpr, threshold_at_target_fpr, actual_target_fpr = compute_tpr_at_target_fpr(
            raw_result["y_true"],
            raw_result["y_probs"],
            target_fpr=target_fpr,
        )
        if threshold_at_target_fpr is None:
            print(f"TPR@{target_fpr * 100:.0f}%FPR: unavailable")
            continue

        print(
            f"TPR@{target_fpr * 100:.0f}%FPR: {target_tpr * 100:.2f}% "
            f"(threshold={threshold_at_target_fpr:.4f}, actual FPR={actual_target_fpr * 100:.2f}%)"
        )
        target_result = summarize_mnw_evaluation(
            raw_result,
            threshold=threshold_at_target_fpr,
            report_title=f"Joint Evaluation Final Report | Target FPR <= {target_fpr * 100:.2f}%",
        )
        report_payload["target_fpr_reports"][f"{target_fpr:.4f}"] = build_report_record(
            target_result,
            mode="target_fpr",
            target_fpr=target_fpr,
            achieved_tpr=target_tpr,
            achieved_fpr=actual_target_fpr,
        )

    results_path = save_test_results(
        result,
        save_path=args.results_save_path,
        output_dir=args.output_dir,
    )
    print(f"Test results saved to: {results_path}")
    reports_path = save_evaluation_reports(
        report_payload,
        save_path=args.reports_save_path,
        output_dir=args.output_dir,
    )
    print(f"Evaluation reports saved to: {reports_path}")
    roc_save_path = resolve_output_path(args.roc_save_path, output_dir=args.output_dir)
    plot_roc_with_target_fpr(
        result["y_true"],
        result["y_probs"],
        target_fpr=args.roc_target_fpr,
        save_path=roc_save_path,
    )
    return result


def build_report_record(result, mode, target_fpr=None, achieved_tpr=None, achieved_fpr=None):
    return {
        "mode": mode,
        "report_title": result["report_title"],
        "threshold": result["threshold"],
        "target_fpr": target_fpr,
        "achieved_tpr": achieved_tpr,
        "achieved_fpr": achieved_fpr,
        "metrics": {
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "fpr": result["fpr"],
            "f1_score": result["f1_score"],
        },
        "confusion_matrix": result["confusion_matrix"],
        "generator_results": result["generator_results"],
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train-causal":
        config = apply_causal_overrides(CausalConfig(), args)
        train_causal(config)
        return

    if args.command == "test-causal":
        config = CausalConfig(output_dir=args.output_dir, token=args.token)
        config.device = resolve_device(args.device)
        model = load_causal_for_mnw(config)
        run_test(model, config, args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
