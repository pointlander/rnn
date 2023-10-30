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
		y := Y[i]
		for k := 0; k < 64; k++ {
			a, b := -1, -1
			if (x>>k)&1 == 1 {
				a = 1
			}
			if (y>>k)&1 == 1 {
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

// Matrix8 is a 8 bit matrix
type Matrix8 struct {
	Cols int
	Rows int
	Data []int8
}

// NewMatrix8 creates a new 8 bit matrix
func NewMatrix8(cols, rows int) Matrix8 {
	m := Matrix8{
		Cols: cols,
		Rows: rows,
		Data: make([]int8, 0, cols*rows),
	}
	return m
}

// Size is the size of the matrix
func (m Matrix8) Size() int {
	return m.Cols * m.Rows
}

func dot8(X, Y []int8) int {
	sum := 0
	for i, x := range X {
		y := Y[i]
		sum += int(x * y)
	}
	return sum
}

// Layer8 is a neural network layer
func Layer8(m Matrix8, n Matrix8, bias Matrix8) Matrix8 {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix8{
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
			z := dot8(mm, nn) + int(bias.Data[inner])
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
