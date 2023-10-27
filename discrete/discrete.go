// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package discrete

// LUT is an 8 entry look up table
type LUT uint8

// FPGA is a field programmable gate array
type FPGA struct {
	MEM []uint8
	LUT []LUT
	ADJ []uint8
}

// NewFPGA creates a new FPGA
func NewFPGA(size int) FPGA {
	return FPGA{
		MEM: make([]uint8, size),
		LUT: make([]LUT, size),
		ADJ: make([]uint8, 3*size*size),
	}
}

// Simulate simulates the FPGA
func (f *FPGA) Simulate() {

}
