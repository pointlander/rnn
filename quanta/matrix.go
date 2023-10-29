// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quanta

import "fmt"

// Matrix is a matrix
type Matrix struct {
	Cols int
	Rows int
	Data []uint64
}

// NewMatrix creates a new matrix
func NewMatrix(cols, rows int) Matrix {
	if (cols*rows)%64 != 0 {
		panic("64 must divide cols")
	}
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]uint64, 0, cols*rows/64),
	}
	return m
}

// Size is the size of the matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

func dot(X, Y []uint64) int {
	sum := 0
	for i, x := range X {
		for k := 0; k < 64; k++ {
			a := -1
			if (x>>k)&1 == 1 {
				a = 1
			}
			b := -1
			if (Y[i]>>k)&1 == 1 {
				b = 1
			}
			sum += a * b
		}
	}
	return sum
}

// Layer is a neural network layer
func Layer(m Matrix, n Matrix, bias Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols / 64
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]uint64, m.Rows*n.Rows/64),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	outer := 0
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		inner := 0
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			b := -1
			if (bias.Data[inner/64]>>(inner%64))&1 == 1 {
				b = 1
			}
			z := dot(mm, nn) + b
			if z > 0 {
				z = 1
			} else {
				z = 0
			}
			o.Data[outer/64] |= uint64(z) << (outer % 64)
			inner++
			outer++
		}
	}
	return o
}
