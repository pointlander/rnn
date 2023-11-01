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

// Instruction is a bf instruction
type Instruction uint8

const (
	// InstructionNoop no operation
	InstructionNoop Instruction = iota
	// InstructionIncrementPointer >
	InstructionIncrementPointer
	// InstructionDecrementPointer <
	InstructionDecrementPointer
	// InstructionIncrement +
	InstructionIncrement
	// InstructionDecrement -
	InstructionDecrement
	// InstructionOutput .
	InstructionOutput
	// InstructionInput ,
	InstructionInput
	// InstructionJumpForward [
	InstructionJumpForward
	// InstructionJumpBack ]
	InstructionJumpBack
	// InstructionNumOps is the number of operations
	InstructionNum
)

// BF is a bf state machine
type BF struct {
	Pointer uint
	Memory  []uint8
	In      uint
	Input   []uint8
	Output  []uint8
	Program []Instruction
}

// String generates the string representation of the program
func (b BF) String() string {
	program := ""
	for _, instruction := range b.Program {
		switch instruction {
		case InstructionIncrementPointer:
			program += ">"
		case InstructionDecrementPointer:
			program += "<"
		case InstructionIncrement:
			program += "+"
		case InstructionDecrement:
			program += "-"
		case InstructionOutput:
			program += "."
		case InstructionInput:
			program += ","
		case InstructionJumpForward:
			program += "["
		case InstructionJumpBack:
			program += "]"
		}
	}
	return program
}

// Compile compiles the program
func Compile(program string, size int) BF {
	input := []byte(program)
	bf := BF{
		Memory: make([]uint8, size),
	}
	for _, symbol := range input {
		switch symbol {
		case '>':
			bf.Program = append(bf.Program, InstructionIncrementPointer)
		case '<':
			bf.Program = append(bf.Program, InstructionDecrementPointer)
		case '+':
			bf.Program = append(bf.Program, InstructionIncrement)
		case '-':
			bf.Program = append(bf.Program, InstructionDecrement)
		case '.':
			bf.Program = append(bf.Program, InstructionOutput)
		case ',':
			bf.Program = append(bf.Program, InstructionInput)
		case '[':
			bf.Program = append(bf.Program, InstructionJumpForward)
		case ']':
			bf.Program = append(bf.Program, InstructionJumpBack)
		}
	}
	return bf
}

// Run the bf program
func (b *BF) Run() {
	length, count := uint(len(b.Memory)), uint(len(b.Input))
	for i := 0; i < len(b.Program); i++ {
		instruction := b.Program[i]
		switch instruction {
		case InstructionNoop:
		case InstructionIncrementPointer:
			b.Pointer++
		case InstructionDecrementPointer:
			b.Pointer--
		case InstructionIncrement:
			b.Memory[b.Pointer%length]++
		case InstructionDecrement:
			b.Memory[b.Pointer%length]--
		case InstructionOutput:
			b.Output = append(b.Output, b.Memory[b.Pointer%length])
		case InstructionInput:
			b.Memory[b.Pointer%length] = b.Input[b.In%count]
			b.In++
		case InstructionJumpForward:
			if b.Memory[b.Pointer%length] != 0 {
				break
			}
			depth := 0
			i++
		ForwardSearch:
			for i < len(b.Program) {
				instruction := b.Program[i]
				switch instruction {
				case InstructionJumpForward:
					depth++
				case InstructionJumpBack:
					if depth == 0 {
						break ForwardSearch
					}
					depth--
				}
				i++
			}
		case InstructionJumpBack:
			if b.Memory[b.Pointer%length] == 0 {
				break
			}
			depth := 0
			i--
		BackSearch:
			for i >= 0 {
				instruction := b.Program[i]
				switch instruction {
				case InstructionJumpForward:
					if depth == 0 {
						break BackSearch
					}
					depth--
				case InstructionJumpBack:
					depth++
				}
				i--
			}
		}
	}
}

// Reset the bf state machine
func (b *BF) Reset() {
	b.Pointer = 0
	for i := range b.Memory {
		b.Memory[i] = 0
	}
	b.In = 0
	b.Output = []uint8{}
}
