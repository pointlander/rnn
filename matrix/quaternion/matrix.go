// Copyright 2023 The Chai Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quaternion

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/num/quat"
)

// Matrix is a quaternion matrix
type Matrix struct {
	Cols   int
	Rows   int
	Data   []quat.Number
	States [][]quat.Number
}

// NewMatrix creates a new quaternion matrix
func NewMatrix(states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]quat.Number, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]quat.Number, states)
		for i := range m.States {
			m.States[i] = make([]quat.Number, cols*rows)
		}
	}
	return m
}

// NewRandMatrix creates a new random quaternion matrix
func NewRandQuatMatrix(rnd *rand.Rand, states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]quat.Number, 0, cols*rows),
	}
	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < cols*rows; i++ {
		m.Data = append(m.Data, quat.Number{
			Real: rnd.NormFloat64() * factor,
			Imag: rnd.NormFloat64() * factor,
			Jmag: rnd.NormFloat64() * factor,
			Kmag: rnd.NormFloat64() * factor,
		})
	}
	if states > 0 {
		m.States = make([][]quat.Number, states)
		for i := range m.States {
			m.States[i] = make([]quat.Number, cols*rows)
		}
	}
	return m
}

// Size is the size of the quaternion matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

func dot(X, Y []quat.Number) quat.Number {
	var sum quat.Number
	for i, x := range X {
		sum = quat.Add(sum, quat.Mul(x, Y[i]))
	}
	return sum
}

// MulT multiplies two quaternion matrices and computes the transpose
func MulT(m Matrix, n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]quat.Number, 0, m.Rows*n.Rows),
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

// Add adds two quaternion matrices
func Add(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]quat.Number, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, quat.Add(value, n.Data[i%lenb]))
	}
	return o
}

// Activation is a quaternion activation function
func Activation(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]quat.Number, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		var v quat.Number
		if value.Real > 0 {
			v.Real = 1
		} else {
			v.Real = -1
		}
		if value.Imag > 0 {
			v.Imag = 1
		} else {
			v.Imag = -1
		}
		if value.Jmag > 0 {
			v.Jmag = 1
		} else {
			v.Jmag = -1
		}
		if value.Kmag > 0 {
			v.Kmag = 1
		} else {
			v.Kmag = -1
		}
		o.Data = append(o.Data, v)
	}
	return o
}

// EverettActivation is the everett quaternion activation function
func EverettActivation(m Matrix) Matrix {
	o := Matrix{
		Cols: 2 * m.Cols,
		Rows: m.Rows,
		Data: make([]quat.Number, 0, 2*m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		rmin, rmax := value.Real, value.Real
		if rmin > 0 {
			rmin = 0
		}
		if rmax < 0 {
			rmax = 0
		}
		imin, imax := value.Imag, value.Imag
		if imin > 0 {
			imin = 0
		}
		if imax < 0 {
			imax = 0
		}
		jmin, jmax := value.Jmag, value.Jmag
		if jmin > 0 {
			jmin = 0
		}
		if jmax < 0 {
			jmax = 0
		}
		kmin, kmax := value.Kmag, value.Kmag
		if kmin > 0 {
			kmin = 0
		}
		if kmax < 0 {
			kmax = 0
		}
		o.Data = append(o.Data, quat.Number{
			Real: rmin,
			Imag: imin,
			Jmag: jmin,
			Kmag: kmin,
		}, quat.Number{
			Real: rmax,
			Imag: imax,
			Jmag: jmax,
			Kmag: kmax,
		})
	}
	return o
}
