#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <iostream> 
#include <chrono>
#include "Params.h"
#include "NBodySeq.h"
#include "BarnesHutCUDA.h"


#define STRINGIFY(X) #X


// Vertex shader
const GLchar* vertexSource =
"#version 130\n"
"in vec2 position;"
"uniform mat4 model;"
"uniform mat4 view;"
"uniform mat4 projection;"
"void main()"
"{"
"    gl_Position = projection * view * model *vec4(position, 0.0, 1.0);"
"}";
// Fragment shader
const GLchar* fragmentSource =
"#version 130\n"
"void main()"
"{"
"    gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);"
"}";

static unsigned int CompileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);
    // TODO: Error handling
    int res;
    glGetShaderiv(id, GL_COMPILE_STATUS, &res);
    if (res == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* mess = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, mess);
        std::cout << "Failed to compile shader: " << id << " - " << (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment") << std::endl;
        std::cout << mess << std::endl;
        glDeleteShader(id);
        return 0;
    }
    return id;
}

static unsigned int createShader(const std::string& vertexShader, const std::string& fragmentShader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

    glAttachShader(program, vs);
    glAttachShader(program, fs);

    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}


void displayDeviceProperties()
{
    // Set up CUDA device 
    cudaDeviceProp properties;

    cudaGetDeviceProperties(&properties, 0);

    int fact = 1024;
    int driverVersion, runtimeVersion;

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    std::cout << "************************************************************************" << std::endl;
    std::cout << "                          GPU Device Properties                         " << std::endl;
    std::cout << "************************************************************************" << std::endl;
    std::cout << "Name:                                    " << properties.name << std::endl;
    std::cout << "CUDA driver/runtime version:             " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << "/" << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
    std::cout << "CUDA compute capabilitiy:                " << properties.major << "." << properties.minor << std::endl;
    std::cout << "Number of multiprocessors:               " << properties.multiProcessorCount << std::endl;
    std::cout << "GPU clock rate:                          " << properties.clockRate / fact << " (MHz)" << std::endl;
    std::cout << "Memory clock rate:                       " << properties.memoryClockRate / fact << " (MHz)" << std::endl;
    std::cout << "Memory bus width:                        " << properties.memoryBusWidth << "-bit" << std::endl;
    std::cout << "Theoretical memory bandwidth:            " << (properties.memoryClockRate / fact * (properties.memoryBusWidth / 8) * 2) / fact << " (GB/s)" << std::endl;
    std::cout << "Device global memory:                    " << properties.totalGlobalMem / (fact * fact) << " (MB)" << std::endl;
    std::cout << "Shared memory per block:                 " << properties.sharedMemPerBlock / fact << " (KB)" << std::endl;
    std::cout << "Constant memory:                         " << properties.totalConstMem / fact << " (KB)" << std::endl;
    std::cout << "Maximum number of threads per block:     " << properties.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum thread dimension:                [" << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[2] << "]" << std::endl;
    std::cout << "Maximum grid size:                       [" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << "]" << std::endl;
    std::cout << "**************************************************************************" << std::endl;
    std::cout << "                                                                          " << std::endl;
    std::cout << "**************************************************************************" << std::endl;
}

void mouseCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS)
    {
        std::cout << "Klik!" << std::endl;
        // CREATE PARTICLE MAYBE?
    }
}

void displayFPS(int frameCount) {
    std::cout << "\r" << "FPS: " << frameCount;
}

int main(int argc, char* argv[])
{
    GLFWwindow* window = 0;

    
    
    if (params.visualize)
    {
        /* Initialize the library */
        if (!glfwInit())
            return -1;
        /* Create a windowed mode window and its OpenGL context */
        window = glfwCreateWindow(980, 980, "NBodySim", NULL, NULL);
        if (!window)
        {
            glfwTerminate();
            return -1;
        }

        /* Make the window's context current */
        glfwMakeContextCurrent(window);

        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
            /* Problem: glewInit failed, something is seriously wrong. */
            fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        }

        fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
        fprintf(stdout, "Status: Using OPENGL %s\n", glGetString(GL_VERSION));

    }
    
    // Parse cmd arguments
    if (argc > 1)
    {
        for (int i = 0; i < argc; i++)
        {
            if (strcmp(argv[i], "-v") == 0)
                params.visualize = true;
            else if (strcmp(argv[i], "-tree") == 0)
                params.display_tree = true;
            else if (strcmp(argv[i], "-num") == 0)
            {
                // Get particle number
                params.num_particles = std::atoi(argv[i + 1]);
                i++;
            }
        }
    }


    // Create simulation
    NBodySeq simulation(params.num_particles);
    BarnesHutCUDA simulation_CUDA(params.num_particles);
    simulation_CUDA.init();

    GLuint vao;
    GLuint vbo;
    unsigned int shader;
    if (params.visualize)
    {
        shader = createShader(vertexSource, fragmentSource);
        glUseProgram(shader);
        


        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);   //generate a buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo);   //make buffer active

           // Specify the layout of the vertex data
        GLint posAttrib = glGetAttribLocation(shader, "position");
        glEnableVertexAttribArray(posAttrib);
        glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);

        glEnable(GL_POINT_SMOOTH);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glPointSize(5.0);


        // model, view, and projection matrices
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::mat4(1.0f);
        // view = glm::rotate(view, float(2*i), glm::vec3(0.0f, 1.0f, 0.0f)); 
        glm::mat4 projection = glm::ortho(-25.0f, 25.0f, -25.0f, 25.0f, -10.0f, 10.0f);

        // link matrices with shader program
        GLint modelLoc = glGetUniformLocation(shader, "model");
        GLint viewLoc = glGetUniformLocation(shader, "view");
        GLint projLoc = glGetUniformLocation(shader, "projection");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    }

    displayDeviceProperties();
    
    //glfwSetMouseButtonCallback(window, mouseCallback);

    int cnt = 0;
    double previousTime = glfwGetTime();
    int frameCount = 0;


    const float* vertices;
    if (_RUN_CUDA) vertices = simulation_CUDA.getOutput();
    else vertices = simulation.getDrawPositions();
    /* Loop until the user closes the window or until all iterations are done */
    int numIters = 1000;
    int iter = 0;
    while (!params.visualize || !glfwWindowShouldClose(window))
    {
        if (iter++ > numIters) break;
        // Measure speed
        //double currentTime = glfwGetTime();
        //frameCount++;
        //// If a second has passed.
        //if (currentTime - previousTime >= 1)
        //{
        //    // Display the frame count here any way you want.
        //    //displayFPS(frameCount);

        //    frameCount = 0;
        //    previousTime = currentTime;
        //}

        // Run simulation
        if (!params.display_times || ++cnt % 200 == 0)
        {
            auto start = std::chrono::high_resolution_clock::now();

            if(BRUTEFORCE) simulation.runBruteForce();
            else simulation.runBarnesHut();
            
            if(_RUN_CUDA) simulation_CUDA.update();

            if (params.display_times) {
                auto stop = std::chrono::high_resolution_clock::now();

                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "Time for 1 iteration:" << duration.count() / 1000 << " milliseconds " << std::endl;
                //std::cout << "NumCalcs: " << simulation.getNumCalcs() << " Num Nodes: " << simulation.getNumNodes() << std::endl;
            }
            cnt = 0;
        }

        if(_RUN_CUDA) vertices = simulation_CUDA.getOutput();

        if (params.visualize)
        {
            glBufferData(GL_ARRAY_BUFFER, 2 * params.num_particles * sizeof(float), vertices, GL_DYNAMIC_DRAW); //copy data to active buffer 

            // Clear the screen to black
            glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
            glClear(GL_COLOR_BUFFER_BIT);

            // Draw points
            glDrawArrays(GL_POINTS, 0, params.num_particles);

            if (params.display_tree) simulation.displayLines();

            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();

            //glDeleteBuffers(1, &vbo);

            //glDeleteVertexArrays(1, &vao);
        }
        
    }
    if (params.visualize)
    {
        glDeleteProgram(shader);
        glfwTerminate();
    }
    return 0;
}