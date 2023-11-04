// Copyright 2023 The RNN Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package discrete

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"

	"github.com/pointlander/rnn/recurrent"
)

const (
	// Size is the number of instructions
	Size = 2 * 1024
	// Window is the window size
	Window = 16
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
	samples := make([]Sample, 1024)
	minLoss := math.MaxFloat64
	done := make(chan bool, 8)
	cpus := runtime.NumCPU()
	inference := func(seed int64, j int) {
		rng := rand.New(rand.NewSource(seed))
		samples[j] = d.Sample(rng)
		samples[j].Run()
		output := samples[j].Output
		/*loss := levenshtein.DistanceForStrings([]rune("Hello World!"), []rune(output),
		levenshtein.DefaultOptions)*/
		target := []byte("Hello World!")
		loss := float64(len(target) - len(output))
		if loss < 0 {
			loss = -loss
		}
		if len(output) > 0 {
			for j, target := range target {
				min, index := math.MaxInt, 0
				for k, symbol := range output {
					diff := int(target) - int(symbol)
					if diff < 0 {
						diff = -diff
					}
					if diff < min {
						min, index = diff, k
					}
				}
				diff := j - index
				if diff < 0 {
					diff = -diff
				}
				loss += float64(diff + min)
			}
		} else {
			loss = math.MaxInt
		}
		samples[j].Loss = float64(loss)
		done <- true
	}
	for e := 0; e < 1024; e++ {
		k, flight := 0, 0
		for j := 0; j < cpus && k < len(samples); j++ {
			go inference(rng.Int63(), k)
			flight++
			k++
		}
		for k < len(samples) {
			<-done
			flight--
			go inference(rng.Int63(), k)
			flight++
			k++
		}
		for flight > 0 {
			<-done
			flight--
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Loss < samples[j].Loss
		})
		min, index := math.MaxFloat64, 0
		for j := 0; j < 64-Window; j++ {
			mean := 0.0
			for k := 0; k < Window; k++ {
				mean += samples[j+k].Loss
			}
			mean /= Window
			stddev := 0.0
			for k := 0; k < Window; k++ {
				diff := mean - samples[j+k].Loss
				stddev += diff * diff
			}
			stddev /= Window
			stddev = math.Sqrt(stddev)
			if stddev < min {
				min, index = stddev, j
			}
		}
		if samples[index].Loss < minLoss {
			minLoss = samples[index].Loss
		} else {
			continue
		}
		fmt.Println(min, index, samples[index].Loss)
		fmt.Println(samples[index].Output)
		fmt.Println(samples[index].String())
		next := Distribution{
			Instructions: make([][]Random, Size),
		}
		for i := range next.Instructions {
			next.Instructions[i] = make([]Random, int(InstructionNum))
		}
		for j := 0; j < Window; j++ {
			for x := 0; x < Size; x++ {
				for y := 0; y < int(InstructionNum); y++ {
					next.Instructions[x][y].Mean += float64(samples[index+j].Instructions.Data[x*int(InstructionNum)+y])
				}
			}
		}
		for j := range next.Instructions {
			for x := range next.Instructions[j] {
				next.Instructions[j][x].Mean /= Window
			}
		}
		for j := 0; j < Window; j++ {
			for x := 0; x < Size; x++ {
				for y := 0; y < int(InstructionNum); y++ {
					diff := next.Instructions[x][y].Mean -
						float64(samples[index+j].Instructions.Data[x*int(InstructionNum)+y])
					next.Instructions[x][y].Stddev += diff * diff
				}
			}
		}
		for j := range next.Instructions {
			for x := range next.Instructions[j] {
				next.Instructions[j][x].Stddev /= Window
				next.Instructions[j][x].Stddev = math.Sqrt(next.Instructions[j][x].Stddev)
			}
		}
		d = next
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
