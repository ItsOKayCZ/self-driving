def set_parameters(data_channel) -> None:
    # Wide - 15
    # Slim - 10
    data_channel.set_int_parameter("roadSize", 15)

    # 0 -> Amazon road
    # 1 -> Black & white road
    data_channel.set_int_parameter("roadColor", 0)

    data_channel.set_bool_parameter("randomBackgroundColor", True)

    data_channel.set_float_parameter("changingBackgroundColorSpeed", 0.75)

    # When the parameter 'backgroundColor' is not set, it is going to generate random colors
    # Values of the channels are <0, 255>
    # data_channel.set_color_parameter('backgroundColor', (255, 0, 0))

    # How well you can see the reflection
    # Default: 1.0
    data_channel.set_float_parameter("reflectionStrength", 0.4)

    # How scaled are the reflections in the X and Y directions
    # Scaling both the numbers up will make the reflections smaller
    # Default: 0.4
    data_channel.set_float_parameter("noiseScaleX", 0.04)
    # Default: 0.05
    data_channel.set_float_parameter("noiseScaleY", 0.05)

    # The animation speed of the reflection
    # Range <0, 1>
    # Default: 0.4
    data_channel.set_float_parameter("noiseSpeed", 0.2)
