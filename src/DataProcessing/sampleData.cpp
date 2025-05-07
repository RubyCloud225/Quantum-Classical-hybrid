 // Save to file
    void saveSamples(const std::vector<Sample>& samples, const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        for (const auto& sample : samples) {
            size_t size = sample.token_embedding.size();
            size_t noise_size = sample.noise.size();
            out.write(reinterpret_cast<const char*>(&size), sizeof(size));
            out.write(reinterpret_cast<const char*>(sample.token_embedding.data()), size * sizeof(double));
            out.write(reinterpret_cast<const char*>(&noise_size), sizeof(noise_size));
            out.write(reinterpret_cast<const char*>(sample.noise.data()), noise_size * sizeof(double));
            out.write(reinterpret_cast<const char*>(&sample.target_value), sizeof(sample.target_value));
            out.write(reinterpret_cast<const char*>(&sample.normalized_noise), sizeof(sample.normalized_noise));
            out.write(reinterpret_cast<const char*>(&sample.density), sizeof(sample.density));
            out.write(reinterpret_cast<const char*>(&sample.nll), sizeof(sample.nll));
            out.write(reinterpret_cast<const char*>(&sample.entropy), sizeof(sample.kl));
        }
        out.close();
    }

    std::vector<sample> loadSamples(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        std::vector<Sample> samples;
        while (in.peek() != EOF) {
            Sample sample;
            size_t size, noise_size;
            in.read(reinterpret_cast<char*>(&size), sizeof(size));
            sample.token_embedding.resize(size);
            in.read(reinterpret_cast<char*>(sample.token_embedding.data()), size * sizeof(double));
            in.read(reinterpret_cast<char*>(&noise_size), sizeof(noise_size));
            sample.noise.resize(noise_size);
            in.read(reinterpret_cast<char*>(sample.noise.data()), noise_size * sizeof(double));
            in.read(reinterpret_cast<char*>(&sample.target_value), sizeof(sample.target_value));
            samples.push_back(sample);
            in.read(reinterpret_cast<char*>(&sample.normalized_noise), sizeof(sample.normalized_noise));
            in.read(reinterpret_cast<char*>(&sample.density), sizeof(sample.density));
            in.read(reinterpret_cast<char*>(&sample.nll), sizeof(sample.nll));
            in.read(reinterpret_cast<char*>(&sample.entropy), sizeof(sample.entropy));
            samples.push_back(sample);
            // Check for read errors
            if (in.fail()) {
                throw std::runtime_error("Error reading sample from file");
            }

        }

        in.close();
        return samples;
    }