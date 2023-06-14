// RUN: mlir-opt %s -split-input-file -affine-loop-tile="cache-size=256 fp-chunk-divisor=32" | FileCheck %s

// CHECK-LABEL: func @simple
func.func @simple(%A : memref<256xf32>, %C : memref<232xf32>) {
  // CHECK: affine.for %{{arg[0-9]+}} = 8 to 240 {
  affine.for %i = 8 to 240 step 1 {
    %0 = affine.load %A[%i - 8] : memref<256xf32>
    %1 = affine.load %A[%i + 2] : memref<256xf32>
    %2 = affine.load %A[%i + 16] : memref<256xf32>
    %3 = arith.addf %0, %1 : f32
    %4 = arith.addf %2, %3 : f32
    affine.store %4, %C[%i] : memref<232xf32>
  }
  return
}

// -----

// CHECK-LABEL: func @complex
func.func @complex(%A : memref<7384xf32, 1>, %B : memref<404416xf32, 1>, %C : memref<9256xf32, 1>) {
  // CHECK: step 13
  affine.for %i = 0 to 13 step 1 {
    // CHECK-NEXT: step 712
    affine.for %j = 0 to 712 step 1 {
      // CHECK-NEXT: step 8
      affine.for %k = 0 to 568 step 1 {
        %0 = affine.load %C[712 * %i + %j] : memref<9256xf32, 1>
        %1 = affine.load %A[568 * %i + %k] : memref<7384xf32, 1>
        %2 = affine.load %B[712 * %k + %j] : memref<404416xf32, 1>
        %3 = arith.addf %0, %1 : f32
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %C[712 * %i + %j] : memref<9256xf32, 1>
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func @complex2
func.func @complex2(%arg0: memref<64x64xf32, 1>, %arg1: memref<32x5x5xf32, 1>, %arg2: memref<32xf32, 1>, %arg3: memref<32x60x60xf32, 1>) {
  // CHECK: step 16
  affine.for %arg6 = 0 to 32 {
    // CHECK: step 60
    affine.for %arg7 = 0 to 60 {
      // CHECK: step 60
      affine.for %arg8 = 0 to 60 {
        // CHECK: step 5
        affine.for %arg9 = 0 to 5 {
          // CHECK: step 5
          affine.for %arg10 = 0 to 5 {
            %0 = affine.load %arg3[%arg6, %arg7, %arg8] : memref<32x60x60xf32, 1>
            %1 = affine.load %arg0[%arg9 + %arg7, %arg10 + %arg8] : memref<64x64xf32, 1>
            %2 = affine.load %arg1[%arg6, %arg9, %arg10] : memref<32x5x5xf32, 1>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %0, %3 : f32
            affine.store %4, %arg3[%arg6, %arg7, %arg8] : memref<32x60x60xf32, 1>
          }
        }
      }
    }
  }
  // CHECK: step 16
  affine.for %arg6 = 0 to 32 {
    // CHECK: step 60
    affine.for %arg7 = 0 to 60 {
      // CHECK: step 60
      affine.for %arg8 = 0 to 60 {
        %0 = affine.load %arg2[%arg6] : memref<32xf32, 1>
        %1 = affine.load %arg3[%arg6, %arg7, %arg8] : memref<32x60x60xf32, 1>
        %2 = arith.maxf %0, %1 : f32
        affine.store %2, %arg2[%arg6] : memref<32xf32, 1>
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: func @fits
func.func @fits(%arg0: memref<32x64xf32, 1>, %arg1: memref<32xf32, 1>, %arg2: memref<64xf32, 1>) {
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: affine.for %{{arg[0-9]+}} = 0 to 64 {
  affine.for %arg3 = 0 to 64 {
    // CHECK: affine.for %{{arg[0-9]+}} = 0 to 32 {
    affine.for %arg4 = 0 to 32 {
      %0 = affine.load %arg2[%arg3] : memref<64xf32, 1>
      %1 = affine.load %arg1[%arg4] : memref<32xf32, 1>
      %2 = affine.load %arg0[%arg4, %arg3] : memref<32x64xf32, 1>
      %3 = arith.addf %1, %2 : f32
      %4 = arith.addf %0, %3 : f32
      affine.store %4, %arg2[%arg3] : memref<64xf32, 1>
    }
  }
  // CHECK: affine.for %{{arg[0-9]+}} = 0 to 64 {
  affine.for %arg3 = 0 to 64 {
    %0 = affine.load %arg2[%arg3] : memref<64xf32, 1>
    %1 = arith.cmpf olt, %0, %cst : f32
    %2 = arith.select %1, %cst, %0 : f32
    affine.store %2, %arg2[%arg3] : memref<64xf32, 1>
  }
  return
}