import numpy as np
from pytorchfi.core import fault_injection
import struct


class BitFlipFI(fault_injection):

    def __init__(self, model, batch_size, fault_location, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [3, 224, 224]
        super().__init__(model, batch_size, input_shape=input_shape, **kwargs)

        self.layer = fault_location[0]
        self.k = fault_location[1]
        self.dim1 = fault_location[2]
        self.dim2 = fault_location[3]
        self.dim3 = fault_location[4]

        self.bit = fault_location[5] if len(fault_location) > 5 else None
        self.value = fault_location[6] if len(fault_location) > 6 else None

        self.golden_value = 0
        self.faulted_value = 0
        self.no_change = False

    def float32_bit_flip(self, param_data, corrupt_index):

        self.golden_value = float(param_data[corrupt_index])

        float_list = []
        a = struct.pack('!f', self.golden_value)
        b = struct.pack('!I', int(2. ** self.bit))
        for ba, bb in zip(a, b):
            float_list.append(ba ^ bb)

        self.faulted_value = struct.unpack('!f', bytes(float_list))[0]

        return self.faulted_value

    def float32_stuck_at(self, param_data, corrupt_index):
        self.golden_value = float(param_data[corrupt_index])

        float_list = []
        a = struct.pack('!f', self.golden_value)
        b = struct.pack('!I', int(2. ** self.bit))
        for ba, bb in zip(a, b):
            if self.value == 1:
                float_list.append(ba | bb)
            else:
                float_list.append(ba & (255 - bb))

        self.faulted_value = struct.unpack('!f', bytes(float_list))[0]

        self.no_change = (self.golden_value == self.faulted_value)

        return self.faulted_value

    def declare_weight_stuck_at(self):
        return self.declare_weight_fi(layer_num=self.layer,
                                      k=self.k,
                                      dim1=self.dim1,
                                      dim2=self.dim2,
                                      dim3=self.dim3,
                                      function=self.float32_stuck_at)

    def declare_weight_bit_flip(self):
        return self.declare_weight_fi(layer_num=self.layer,
                                      k=self.k,
                                      dim1=self.dim1,
                                      dim2=self.dim2,
                                      dim3=self.dim3,
                                      function=self.float32_bit_flip)