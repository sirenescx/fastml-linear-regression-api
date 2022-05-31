import grpc
from concurrent import futures

from fast_ml_linear_regression_models.proto.compiled import linear_regression_pb2_grpc
from fast_ml_linear_regression_models.services.linear_regressor import LinearRegressionService


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    linear_regression_pb2_grpc.add_LinearRegressionServiceServicer_to_server(
        LinearRegressionService(), server
    )
    server.add_insecure_port('[::]:84')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
