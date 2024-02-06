package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	input := mat.NewDense(1, 3, nil)
	input.Set(0, 0, 8.5)
	input.Set(0, 1, 0.65)
	input.Set(0, 2, 1.2)
	arr := make([]float64, 9)
	arr = []float64{0.1, 0.1, -0.3, 0.1, 0.2, 0.0, 0.0, 1.3, 0.1}
	weights := mat.NewDense(3, 3, arr)
	D := mat.NewDense(1, 3, nil)
	D.Product(input, weights.T())
	formatted := mat.Formatted(D, mat.Prefix(""), mat.Squeeze())
	fmt.Println(formatted)
}
