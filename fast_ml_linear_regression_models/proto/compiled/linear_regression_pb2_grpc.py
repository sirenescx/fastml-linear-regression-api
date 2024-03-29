# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import linear_regression_pb2 as linear__regression__pb2


class LinearRegressionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Train = channel.unary_unary(
                '/fast_ml_linear_regression.LinearRegressionService/Train',
                request_serializer=linear__regression__pb2.TrainingRequest.SerializeToString,
                response_deserializer=linear__regression__pb2.TrainingResponse.FromString,
                )
        self.Predict = channel.unary_unary(
                '/fast_ml_linear_regression.LinearRegressionService/Predict',
                request_serializer=linear__regression__pb2.PredictionRequest.SerializeToString,
                response_deserializer=linear__regression__pb2.PredictionResponse.FromString,
                )


class LinearRegressionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Train(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LinearRegressionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Train': grpc.unary_unary_rpc_method_handler(
                    servicer.Train,
                    request_deserializer=linear__regression__pb2.TrainingRequest.FromString,
                    response_serializer=linear__regression__pb2.TrainingResponse.SerializeToString,
            ),
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=linear__regression__pb2.PredictionRequest.FromString,
                    response_serializer=linear__regression__pb2.PredictionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'fast_ml_linear_regression.LinearRegressionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LinearRegressionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Train(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fast_ml_linear_regression.LinearRegressionService/Train',
            linear__regression__pb2.TrainingRequest.SerializeToString,
            linear__regression__pb2.TrainingResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/fast_ml_linear_regression.LinearRegressionService/Predict',
            linear__regression__pb2.PredictionRequest.SerializeToString,
            linear__regression__pb2.PredictionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
