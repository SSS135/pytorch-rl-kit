from .cnn_actors import CNNActor
from .utils import image_to_float
from ..common.qrnn import DenseQRNN


class CNN_QRNNActor(CNNActor):
    def __init__(self, *args, qrnn_hidden_size=512, qrnn_layers=2, qrnn_norm=None, **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_sizes: List of hidden layers sizes
            activation: Activation function
        """
        super().__init__(*args, **kwargs)
        self.qrnn_hidden_size = qrnn_hidden_size
        self.qrnn_layers = qrnn_layers
        # self.linear = self.make_layer(nn.Linear(1024, 512))
        self.qrnn = DenseQRNN(self.linear[0].in_features, qrnn_hidden_size, qrnn_layers, norm=qrnn_norm)
        del self.linear
        self.hidden_code_size = qrnn_hidden_size
        self._init_heads(self.hidden_code_size)
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        seq_len, batch_len = input.shape[:2]
        input = input.reshape(seq_len * batch_len, *input.shape[2:])

        input = image_to_float(input)
        x = self._extract_features(input)
        x = x.view(seq_len, batch_len, -1)
        x, next_memory = self.qrnn(x, memory, done_flags)

        head = self._run_heads(x)
        head.hidden_code = x

        if self.do_log:
            self.logger.add_histogram('conv linear', x, self._step)

        return head, next_memory
