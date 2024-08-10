import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op

class SolOp(Op):
    def __init__(self, sol_op_jax_jitted, vjp_sol_op):
        self.sol_op_jax_jitted = sol_op_jax_jitted
        self.vjp_sol_op = vjp_sol_op

    def make_node(self, *inputs):
        # Convert our inputs to symbolic variables
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        # Assume the output to always be a float64 matrix
        outputs = [pt.matrix()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        result = self.sol_op_jax_jitted(*inputs)
        outputs[0][0] = np.asarray(result, dtype="float64")
        
    def grad(self, inputs, output_grads):
        (gz,) = output_grads
        return self.vjp_sol_op(inputs, gz)
    
class VJPSolOp(Op):
    def __init__(self, vjp_sol_op_jax_jitted):
        self.vjp_sol_op_jax_jitted = vjp_sol_op_jax_jitted

    def make_node(self, inputs, gz):
        inputs = [pt.as_tensor_variable(inp) for inp in inputs] 
        inputs += [pt.as_tensor_variable(gz)]
        outputs = [inputs[i].type() for i in range(len(inputs)-1)]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        *params, gz = inputs
        result = self.vjp_sol_op_jax_jitted(gz, *params)
        for i, res in enumerate(result):
            outputs[i][0] = np.asarray(res, dtype="float64")