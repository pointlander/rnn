// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package complex

import (
	"fmt"
	"math"
	"math/rand"
)

// Matrix is a complex matrix
type Matrix struct {
	Cols   int
	Rows   int
	Data   []complex128
	States [][]complex128
}

// NewMatrix creates a new complex matrix
func NewMatrix(states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]complex128, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]complex128, states)
		for i := range m.States {
			m.States[i] = make([]complex128, cols*rows)
		}
	}
	return m
}

// NewRandMatrix creates a new random complex matrix
func NewRandMatrix(rnd *rand.Rand, states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]complex128, 0, cols*rows),
	}
	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < cols*rows; i++ {
		m.Data = append(m.Data, complex(rnd.NormFloat64()*factor, rnd.NormFloat64()*factor))
	}
	if states > 0 {
		m.States = make([][]complex128, states)
		for i := range m.States {
			m.States[i] = make([]complex128, cols*rows)
		}
	}
	return m
}

// Size is the size of the complex matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

func dot(X, Y []complex128) complex128 {
	var sum complex128
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}

// MulT multiplies two complex matrices and computes the transpose
func MulT(m Matrix, n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]complex128, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// Add adds two complex matrices
func Add(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex128, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Activation is a complex activation function
func Activation(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex128, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		var v complex128
		if real(value) > 0 {
			v = 1
		} else {
			v = -1
		}
		if imag(value) > 0 {
			v += 1i
		} else {
			v += -1i
		}
		o.Data = append(o.Data, v)
	}
	return o
}

// EverettActivation is the everett complex activation function
func EverettActivation(m Matrix) Matrix {
	o := Matrix{
		Cols: 2 * m.Cols,
		Rows: m.Rows,
		Data: make([]complex128, 0, 2*m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		rmin, rmax := real(value), real(value)
		if rmin > 0 {
			rmin = 0
		}
		if rmax < 0 {
			rmax = 0
		}
		imin, imax := imag(value), imag(value)
		if imin > 0 {
			imin = 0
		}
		if imax < 0 {
			imax = 0
		}
		o.Data = append(o.Data, complex(rmin, imin), complex(rmax, imax))
	}
	return o
}

// TaylorSoftmax is the taylor softmax
// https://arxiv.org/abs/1511.05042
func TaylorSoftmax(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]complex128, 0, m.Cols*m.Rows),
	}
	var sum complex128
	columns, lenm := m.Cols, len(m.Data)
	for i := 0; i < lenm; i += columns {
		nn := m.Data[i : i+columns]
		for _, v := range nn {
			sum += 1 + v + v*v/2
		}
	}
	for i := 0; i < lenm; i += columns {
		nn := m.Data[i : i+columns]
		for _, v := range nn {
			o.Data = append(o.Data, (1+v+v*v/2)/sum)
		}
	}
	return o
}
