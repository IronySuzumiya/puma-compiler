#include <assert.h>
#include <string>
#include <vector>

#include "puma.h"
#include "conv-layer.h"
#include "fully-connected-layer.h"

void isolated_fully_connected_layer(Model model, std::string layerName, unsigned int in_size, unsigned int out_size) {

    // Input vector
    auto in = InputVector::create(model, "in", in_size);

    // Output vector
    auto out = OutputVector::create(model, "out", out_size);

    // Layer
    out = fully_connected_layer(model, layerName, in_size, out_size, in);

}

int main() {

    Model model = Model::create("test");

    unsigned int in_size_x = 2;
    unsigned int in_size_y = 2;
    unsigned int in_channels = 2;
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);

    unsigned int k_size_x1 = 2;
    unsigned int k_size_y1 = 2;
    unsigned int in_size_x1 = 2;
    unsigned int in_size_y1 = 2;
    unsigned int in_channels1 = 2;
    unsigned int out_channels1 = 2;

    unsigned int out_size_x = 2;
    unsigned int out_size_y = 2;
    unsigned int out_channels = 2;
    auto out_stream = OutputImagePixelStream::create(model, "out_stream", out_size_x, out_size_y, out_channels);

    unsigned int in_size2 = 8;
    unsigned int out_size2 = 4;

    auto out1 = conv_layer(model, "layer" + std::to_string(1), k_size_x1, k_size_y1, in_size_x1, in_size_y1, in_channels1, out_channels1, in_stream);
    out_stream = out1;
    isolated_fully_connected_layer(model, "layer" + std::to_string(2), in_size2, out_size2);

    model.compile();

    model.destroy();

    return 0;
}
