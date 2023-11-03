// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package discrete

import (
	"fmt"
	"math/rand"

	"github.com/pointlander/rnn/recurrent"
	"github.com/texttheater/golang-levenshtein/levenshtein"
)

const (
	// Size is the number of instructions
	Size = 32
)

// Random is a random variable
type Random struct {
	Mean   float64
	Stddev float64
}

// Distribution is a distribution of BF machines
type Distribution struct {
	Instructions [][]Random
}

// NewDistribution is a new distribution of BF machines
func NewDistribution(rng *rand.Rand) Distribution {
	instructions := make([][]Random, Size)
	for i := range instructions {
		for j := 0; j < int(InstructionNum); j++ {
			instructions[i] = append(instructions[i], Random{
				Mean:   rng.NormFloat64(),
				Stddev: rng.NormFloat64(),
			})
		}
	}
	return Distribution{
		Instructions: instructions,
	}
}

// Sample is a BF machine sample
type Sample struct {
	Instructions recurrent.Matrix32
	BF
	Loss float64
}

// Sample samples the distribution
func (d Distribution) Sample(rng *rand.Rand) Sample {
	instructions := recurrent.NewMatrix32(0, int(InstructionNum), Size)
	for i := 0; i < Size; i++ {
		for j := 0; j < int(InstructionNum); j++ {
			random := d.Instructions[i][j]
			instructions.Data = append(instructions.Data,
				float32(rng.NormFloat64()*random.Stddev+random.Mean))
		}
	}
	n := recurrent.Normalize32(instructions)
	y := recurrent.SelfAttention32(n, n, n)
	program := make([]Instruction, Size)
	for i := 0; i < Size; i++ {
		max, instruction := float32(0.0), 0
		for j := 0; j < y.Cols; j++ {
			if y.Data[i*y.Cols+j] > max {
				max, instruction = y.Data[i*y.Cols+j], j
			}
		}
		program[i] = Instruction(instruction)
	}
	return Sample{
		Instructions: instructions,
		BF: BF{
			Memory:  make([]uint8, 30000),
			Program: program,
		},
	}
}

// Learn learns a BF program
func Learn() {
	rng := rand.New(rand.NewSource(1))
	d := NewDistribution(rng)
	for i := 0; i < 128; i++ {
		sample := d.Sample(rng)
		sample.Run()
		output := string(sample.Output)
		loss := levenshtein.DistanceForStrings([]rune("Hello World!"), []rune(output),
			levenshtein.DefaultOptions)
		fmt.Println(loss)
	}
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
	length, count, cycles := uint(len(b.Memory)), uint(len(b.Input)), 0
	for i := 0; i < len(b.Program) && cycles < 8*1024; i++ {
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
			if count <= 0 {
				break
			}
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
		cycles++
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
