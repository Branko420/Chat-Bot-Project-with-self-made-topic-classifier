defmodule AiProjectWeb.PageController do
  use AiProjectWeb, :controller

  def home(conn, _params) do
    # The home page is often custom made,
    # so skip the default app layout.
    render(conn, :home, layout: false)
  end
  def chat(conn, _params)do
    render(conn, "chat.html")
  end
end
