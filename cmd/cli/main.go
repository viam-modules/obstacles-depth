package main

import (
	"context"
	"obstaclesdepth"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/vision"
)

func main() {
	err := realMain()
	if err != nil {
		panic(err)
	}
}

func realMain() error {
	ctx := context.Background()
	logger := logging.NewLogger("cli")

	deps := resource.Dependencies{}
	// can load these from a remote machine if you need

	cfg := obstaclesdepth.Config{}

	thing, err := obstaclesdepth.NewObstaclesDepth(ctx, deps, vision.Named("foo"), &cfg, logger)
	if err != nil {
		return err
	}
	defer thing.Close(ctx)

	return nil
}
