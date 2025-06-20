package main

import (
	"go.viam.com/rdk/module"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/vision"

	obstaclesdepth "obstaclesdepth/obstacles-depth"
)

func main() {
	module.ModularMain(resource.APIModel{
		API:   vision.API,
		Model: obstaclesdepth.Model,
	})
}
