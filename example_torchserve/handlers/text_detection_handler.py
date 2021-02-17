"""
Defines example_torchserve handler for text detection
"""
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        super(ModelHandler, self).__init__()

        self._min_size = 20
        self._text_threshold = 0.7
        self._low_text = (0.4,)
        self._link_threshold = 0.4
        self._canvas_size = 2560
        self._mag_ratio = 1.0
        self._slope_ths = 0.1
        self._ycenter_ths = 0.5
        self._height_ths = 0.5
        self._width_ths = 0.5
        self._add_margin = 0.1

        self._ratio_h = 0
        self._ratio_w = 0

        self._transform = transforms.Compose(
            [
                transforms.Resize(self._canvas_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, data):
        """
        Transform raw input into model input data.

        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        transformed_data = self._transform(data)
        transformed_data.to(self.device)

        return transformed_data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        y, _ = inference_output

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        return score_text, score_link

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
