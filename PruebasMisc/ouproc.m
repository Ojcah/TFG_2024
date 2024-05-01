## Ornstein-Uhlenbeck Process (OU Process)

1;

function [x,y] = ou_process(dt, theta, sigma, x0, T, beta)
  % This function generates an Ornstein-Uhlenbeck process.
  % x is the original result of the OU process
  % y is a smoothed version

  % Parameters:
  %   dt: Time step
  %   theta: Mean reversion rate
  %   sigma: Noise strength (standard deviation)
  %   x0: Initial state
  %   T: Total simulation time
  %   beta: smoothing factor for y

  % Number of steps
  N = floor(T / dt);
  
  % Initialize process output
  x = zeros(1, N);
  x(1) = x0;

  y = x;
  
  k1=exp(-theta*dt);
  k2=sigma*sqrt(dt);
  
  % Generate OU process
  for i = 2:N
    % Update state using Euler-Maruyama method
    x(i) = x(i-1) * k1 + k2 * randn(1); # original
    y(i) = beta*y(i-1) + (1-beta)*x(i); # smoothed
  end
end

% Example usage:
dt = 0.02;   % Time step (1/50)
theta = 1;   % Mean reversion rate
sigma = 0.2; % Noise strength
x0 = 0.5;   % Initial state
T = 10;      % Total simulation time
beta = power(0.01,1/25); % reach 0.01 after 25 steps

% Generate OU process data
[orig,smoothed] = ou_process(dt, theta, sigma, x0, T,beta);

                                % Plot the generated data
t=linspace(0,T,length(orig));
hold off;plot(t,orig,";OU original;","linewidth",2);
hold on;plot(t,smoothed,";Filtered;","linewidth",2);

xlabel('Time');
ylabel('Process Value');
grid on
