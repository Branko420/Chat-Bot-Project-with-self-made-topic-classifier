defmodule AiProject.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      AiProjectWeb.Telemetry,
      {DNSCluster, query: Application.get_env(:ai_project, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: AiProject.PubSub},
      # Start the Finch HTTP client for sending emails
      {Finch, name: AiProject.Finch},
      # Start a worker by calling: AiProject.Worker.start_link(arg)
      # {AiProject.Worker, arg},
      # Start to serve requests, typically the last entry
      AiProjectWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: AiProject.Supervisor]
    Supervisor.start_link(children, opts)
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    AiProjectWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
