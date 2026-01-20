function [v] = conv_hull_eig(eigvals)

x = real(eigvals);
y = imag(eigvals);
idx = convhull(x, y);
v = [x(idx), y(idx)];

end