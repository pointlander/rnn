// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64
// +build amd64

package recurrent

import (
	"github.com/ziutek/blas"
)

func dot(X, Y []float64) float64 {
	return blas.Ddot(len(X), X, 1, Y, 1)
}

func dot32(X, Y []float32) float32 {
	return blas.Sdot(len(X), X, 1, Y, 1)
}
