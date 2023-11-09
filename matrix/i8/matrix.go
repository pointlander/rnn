// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package i8

import "fmt"

// Matrix8 is a 8 bit matrix
type Matrix struct {
	Cols int
	Rows int
	Data []int8
}

// NewMatrix8 creates a new 8 bit matrix
func NewMatrix(cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]int8, 0, cols*rows),
	}
	return m
}

// Size is the size of the matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

func dot(X, Y []int8) int {
	sum := 0
	for i, x := range X {
		y := Y[i]
		sum += int(x * y)
	}
	return sum
}

// Layer is a neural network layer
func Layer(m Matrix, n Matrix, bias Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]int8, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		inner := 0
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			z := dot(mm, nn) + int(bias.Data[inner])
			if z > 0 {
				z = 1
			} else {
				z = -1
			}
			o.Data = append(o.Data, int8(z))
			inner++
		}
	}
	return o
}
