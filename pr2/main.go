package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	var pred float64 = 0.8
	var Alpha float64 = 0.1
	input := mat.NewDense(1, 1, nil)
	input.Set(0, 0, 2.0)
	weight := mat.NewDense(1, 1, nil)
	weight.Set(0, 0, 0.5)
	D := mat.NewDense(1, 1, nil)
	for i := 0; i < 1000; i++ {
		D.Product(input, weight.T())
		arr := D.RawMatrix().Data
		err := (arr[0] - pred) * (arr[0] - pred)
		weight.Set(0, 0, (weight.RawMatrix().Data[0] - (arr[0]-pred)*input.RawMatrix().Data[0]*Alpha))
		fmt.Println(i, "Error: ", err, "Pred", arr[0])
	}
}

func D() {
	var pred float64 = 0.8
	var Alpha float64 = 0.1
	var input float64 = 2
	var weight float64 = 0.5
	for i := 0; i < 1000; i++ {
		m := (input * weight)
		err := (m - pred) * (m - pred)
		weight -= (m - pred) * Alpha * input
		fmt.Println(i, "Error:", err, "Pred:", m)
	}
}
