#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = N; i < N+VECTOR_WIDTH; i++) 
  {
    values[i] = 0;
  }
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    if (i + VECTOR_WIDTH >= N) {
      maskAll = _pp_init_ones(N-i);
    } else {
      maskAll = _pp_init_ones();
    }

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float valuesVec = _pp_vset_float(0.0f);
  __pp_vec_float outputVec = _pp_vset_float(0.0f);
  __pp_vec_int exponentsVec = _pp_vset_int(0);
  __pp_vec_int countVec = _pp_vset_int(0);

  __pp_vec_int zerosInt = _pp_vset_int(0);
  __pp_vec_int onesInt = _pp_vset_int(1);
  __pp_vec_float onesFloat = _pp_vset_float(1.0f);
  __pp_vec_float clampValueFloat = _pp_vset_float(9.999999f);

  __pp_mask maskAll, maskExpIsZero, maskExpNotZero, maskCount, maskClamp;

  for (int i = N; i < N+VECTOR_WIDTH; i++)
  {
    values[i] = 0.0f;
    exponents[i] = 1;
  }

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    maskExpIsZero = _pp_init_ones(0);
    maskCount = _pp_init_ones(0);
    maskClamp = _pp_init_ones(0);

    _pp_vload_float(valuesVec, values+i, maskAll); 
    _pp_vload_int(exponentsVec, exponents+i, maskAll); 

    _pp_veq_int(maskExpIsZero, exponentsVec, zerosInt, maskAll); 
    _pp_vstore_float(output+i, onesFloat, maskExpIsZero); 
    
    maskExpNotZero = _pp_mask_not(maskExpIsZero); 
    _pp_vmove_float(outputVec, valuesVec, maskExpNotZero); 
    _pp_vsub_int(countVec, exponentsVec, onesInt, maskExpNotZero); 

    _pp_vgt_int(maskCount, countVec, zerosInt, maskExpNotZero);
    while(_pp_cntbits(maskCount))
    {
      _pp_vmult_float(outputVec, outputVec, valuesVec, maskCount);
      _pp_vsub_int(countVec, countVec, onesInt, maskCount);
      _pp_vgt_int(maskCount, countVec, zerosInt, maskExpNotZero);
    }

    _pp_vgt_float(maskClamp, outputVec, clampValueFloat, maskExpNotZero);
    _pp_vset_float(outputVec, 9.999999f, maskClamp);
    _pp_vstore_float(output+i, outputVec, maskExpNotZero);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float valuesVec = _pp_vset_float(0.0f);
  __pp_vec_float outputVec = _pp_vset_float(0.0f);
  __pp_mask maskAll;
  
  float *result = new float[VECTOR_WIDTH];
  int shift = VECTOR_WIDTH;
  
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    _pp_vload_float(valuesVec, values+i, maskAll);
    _pp_vadd_float(outputVec, outputVec, valuesVec, maskAll);
  }

  while (shift > 1)
  {
    _pp_hadd_float(outputVec, outputVec);
    _pp_interleave_float(outputVec, outputVec);
    shift /= 2;
  }
  _pp_vstore_float(result, outputVec, maskAll);

  return result[0];
}