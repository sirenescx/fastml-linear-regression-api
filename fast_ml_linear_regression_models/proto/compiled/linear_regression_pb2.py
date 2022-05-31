# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: linear_regression.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17linear_regression.proto\x12\x19\x66\x61st_ml_linear_regression\"\xb9\x01\n\x0fTrainingRequest\x12\x11\n\talgorithm\x18\x01 \x01(\t\x12\x10\n\x08\x66ilepath\x18\x02 \x01(\t\x12N\n\nparameters\x18\x03 \x03(\x0b\x32:.fast_ml_linear_regression.TrainingRequest.ParametersEntry\x1a\x31\n\x0fParametersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\")\n\x10TrainingResponse\x12\x15\n\rconfiguration\x18\x01 \x01(\t\"8\n\x11PredictionRequest\x12\x11\n\talgorithm\x18\x01 \x01(\t\x12\x10\n\x08\x66ilepath\x18\x02 \x01(\t\"%\n\x12PredictionResponse\x12\x0f\n\x07message\x18\x01 \x01(\t2\xe3\x01\n\x17LinearRegressionService\x12`\n\x05Train\x12*.fast_ml_linear_regression.TrainingRequest\x1a+.fast_ml_linear_regression.TrainingResponse\x12\x66\n\x07Predict\x12,.fast_ml_linear_regression.PredictionRequest\x1a-.fast_ml_linear_regression.PredictionResponseBv\n\x19\x63om.fastmllinearrgressionB\x1b\x46\x61stMLLinearRegressionProtoP\x01Z\x19\x66\x61stmllinearregressionspb\xaa\x02\x1e\x46\x61st.ML.Linear.Regression.Grpcb\x06proto3')



_TRAININGREQUEST = DESCRIPTOR.message_types_by_name['TrainingRequest']
_TRAININGREQUEST_PARAMETERSENTRY = _TRAININGREQUEST.nested_types_by_name['ParametersEntry']
_TRAININGRESPONSE = DESCRIPTOR.message_types_by_name['TrainingResponse']
_PREDICTIONREQUEST = DESCRIPTOR.message_types_by_name['PredictionRequest']
_PREDICTIONRESPONSE = DESCRIPTOR.message_types_by_name['PredictionResponse']
TrainingRequest = _reflection.GeneratedProtocolMessageType('TrainingRequest', (_message.Message,), {

  'ParametersEntry' : _reflection.GeneratedProtocolMessageType('ParametersEntry', (_message.Message,), {
    'DESCRIPTOR' : _TRAININGREQUEST_PARAMETERSENTRY,
    '__module__' : 'linear_regression_pb2'
    # @@protoc_insertion_point(class_scope:fast_ml_linear_regression.TrainingRequest.ParametersEntry)
    })
  ,
  'DESCRIPTOR' : _TRAININGREQUEST,
  '__module__' : 'linear_regression_pb2'
  # @@protoc_insertion_point(class_scope:fast_ml_linear_regression.TrainingRequest)
  })
_sym_db.RegisterMessage(TrainingRequest)
_sym_db.RegisterMessage(TrainingRequest.ParametersEntry)

TrainingResponse = _reflection.GeneratedProtocolMessageType('TrainingResponse', (_message.Message,), {
  'DESCRIPTOR' : _TRAININGRESPONSE,
  '__module__' : 'linear_regression_pb2'
  # @@protoc_insertion_point(class_scope:fast_ml_linear_regression.TrainingResponse)
  })
_sym_db.RegisterMessage(TrainingResponse)

PredictionRequest = _reflection.GeneratedProtocolMessageType('PredictionRequest', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTIONREQUEST,
  '__module__' : 'linear_regression_pb2'
  # @@protoc_insertion_point(class_scope:fast_ml_linear_regression.PredictionRequest)
  })
_sym_db.RegisterMessage(PredictionRequest)

PredictionResponse = _reflection.GeneratedProtocolMessageType('PredictionResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREDICTIONRESPONSE,
  '__module__' : 'linear_regression_pb2'
  # @@protoc_insertion_point(class_scope:fast_ml_linear_regression.PredictionResponse)
  })
_sym_db.RegisterMessage(PredictionResponse)

_LINEARREGRESSIONSERVICE = DESCRIPTOR.services_by_name['LinearRegressionService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\031com.fastmllinearrgressionB\033FastMLLinearRegressionProtoP\001Z\031fastmllinearregressionspb\252\002\036Fast.ML.Linear.Regression.Grpc'
  _TRAININGREQUEST_PARAMETERSENTRY._options = None
  _TRAININGREQUEST_PARAMETERSENTRY._serialized_options = b'8\001'
  _TRAININGREQUEST._serialized_start=55
  _TRAININGREQUEST._serialized_end=240
  _TRAININGREQUEST_PARAMETERSENTRY._serialized_start=191
  _TRAININGREQUEST_PARAMETERSENTRY._serialized_end=240
  _TRAININGRESPONSE._serialized_start=242
  _TRAININGRESPONSE._serialized_end=283
  _PREDICTIONREQUEST._serialized_start=285
  _PREDICTIONREQUEST._serialized_end=341
  _PREDICTIONRESPONSE._serialized_start=343
  _PREDICTIONRESPONSE._serialized_end=380
  _LINEARREGRESSIONSERVICE._serialized_start=383
  _LINEARREGRESSIONSERVICE._serialized_end=610
# @@protoc_insertion_point(module_scope)
