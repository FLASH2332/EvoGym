#include <C:\Users\jayad\Desktop\evorobotics\Scripts\evogym\evogym\simulator\externals\glfw/include/GLFW/glfw3.h>
#include <stdio.h>

int main() {
    if (!glfwInit()) {
        printf("Error initializing GLFW\n");
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(800, 600, "Test", NULL, NULL);
    if (!window) {
        printf("Error creating window\n");
        glfwTerminate();
        return -1;
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
