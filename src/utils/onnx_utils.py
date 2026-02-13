import logging
import os

import onnx
import onnxruntime
import torch
import torch.onnx

logger = logging.getLogger(__name__)


def export_to_onnx(
    model, dummy_input, onnx_path, input_names, output_names, dynamic_axes
):

    model.eval()
    logger.info(f"Exporting model to {onnx_path}...")

    dirname = os.path.dirname(onnx_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        logger.info("Model exported successfully.")

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verified.")

    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise


def create_onnx_session(onnx_path, providers=None):

    if providers is None:
        providers = ["CPUExecutionProvider"]

    logger.info(f"Loading ONNX model from {onnx_path} with providers {providers}")
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    return session


def quantize_onnx_model(onnx_path, quantized_model_path):

    from onnxruntime.quantization import QuantType, quantize_dynamic

    logger.info(f"Quantizing model {onnx_path} to {quantized_model_path}...")
    try:
        quantize_dynamic(onnx_path, quantized_model_path, weight_type=QuantType.QUInt8)
        logger.info("Quantization complete.")
    except Exception as e:
        logger.error(f"Failed to quantize model: {e}")
        raise
