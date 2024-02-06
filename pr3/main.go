package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	pred := make([]float64, 3)
	pred[0] = 0.8
	pred[1] = 0.6
	pred[2] = 0.55
	input := make([]float64, 3)
	input[0] = 0.3
	input[1] = 0.2
	input[2] = 0.8
	b1 := make([]float64, 0)
	b2 := make([]float64, 0)
	for i := 0; i < 6; i++ {
		b1 = append(b1, 0.1)
	}
	for i := 0; i < 6; i++ {
		b2 = append(b2, 0.1)
	}
	var Alpha float64 = 0.1
	pred_goal := mat.NewDense(1, 3, pred)
	input_mat := mat.NewDense(1, 3, input)
	input_weight := mat.NewDense(2, 3, b1)
	predict_weight := mat.NewDense(3, 2, b2)
	prediction := mat.NewDense(1, 3, nil)
	E12 := mat.NewDense(1, 2, nil)
	err := mat.NewDense(1, 3, nil)
	pred_delta := mat.NewDense(3, 2, nil)
	input_delta := mat.NewDense(2, 3, nil)
	weight_gradient := mat.NewDense(1, 2, nil)
	for i := 0; i < 1000; i++ {
		E12.Product(input_mat, input_weight.T())
		prediction.Product(E12, predict_weight.T())
		err.Sub(prediction, pred_goal)
		pred_delta.Product(err.T(), E12)
		pred_delta.Scale(Alpha, pred_delta)
		predict_weight.Sub(predict_weight, pred_delta)
		weight_gradient.Product(err, predict_weight)
		input_delta.Product(weight_gradient.T(), input_mat)
		input_delta.Scale(Alpha, input_delta)
		input_weight.Sub(input_weight, input_delta)
		fmt.Println(i, "Errors:", err.RawMatrix().Data[0]*err.RawMatrix().Data[0], err.RawMatrix().Data[1]*err.RawMatrix().Data[1], err.RawMatrix().Data[2]*err.RawMatrix().Data[2])
	}
	E12.Product(input_mat, input_weight.T())
	prediction.Product(E12, predict_weight.T())
	fmt.Println(prediction.RawMatrix().Data[0], prediction.RawMatrix().Data[1], prediction.RawMatrix().Data[2])
}
