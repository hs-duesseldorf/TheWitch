// Copyright 2022 Chris Ringenberg https://www.ringenberg.dev/

#include "Websocket.h"

#include "IWebSocket.h"

namespace
{
	void CopySocketBytes(TArray<uint8>& OutBytes, const void* Data, SIZE_T Size)
	{
		const int32 SafeSize = static_cast<int32>(FMath::Min<SIZE_T>(Size, MAX_int32));
		OutBytes.SetNumUninitialized(SafeSize);

		if (SafeSize > 0)
		{
			FMemory::Memcpy(OutBytes.GetData(), Data, SafeSize);
		}
	}
}

void UWebSocket::InitWebSocket(TSharedPtr<IWebSocket> InWebSocket)
{
	InternalWebSocket = InWebSocket;

	if (!InternalWebSocket.IsValid())
	{
		return;
	}

	InternalWebSocket->OnConnected().AddUObject(this, &ThisClass::OnWebSocketConnected_Internal);
	InternalWebSocket->OnConnectionError().AddUObject(this, &ThisClass::OnWebSocketConnectionError_Internal);
	InternalWebSocket->OnClosed().AddUObject(this, &ThisClass::OnWebSocketClosed_Internal);
	InternalWebSocket->OnMessage().AddUObject(this, &ThisClass::OnWebSocketMessageReceived_Internal);
	InternalWebSocket->OnBinaryMessage().AddUObject(this, &ThisClass::OnWebSocketBinaryMessageReceived_Internal);
	InternalWebSocket->OnMessageSent().AddUObject(this, &ThisClass::OnWebSocketMessageSent_Internal);
}

void UWebSocket::Connect()
{
	if (InternalWebSocket.IsValid())
	{
		InternalWebSocket->Connect();
	}
}

void UWebSocket::Close(int32 StatusCode, const FString& Reason)
{
	if (InternalWebSocket.IsValid())
	{
		InternalWebSocket->Close(StatusCode, Reason);
	}
}

bool UWebSocket::IsConnected() const
{
	return InternalWebSocket.IsValid() && InternalWebSocket->IsConnected();
}

void UWebSocket::SendMessage(const FString& Message)
{
	if (InternalWebSocket.IsValid())
	{
		InternalWebSocket->Send(Message);
	}
}

void UWebSocket::SendBinaryMessage(const TArray<uint8>& Data)
{
	if (InternalWebSocket.IsValid() && !Data.IsEmpty())
	{
		InternalWebSocket->Send(Data.GetData(), Data.Num(), true);
	}
}

void UWebSocket::BeginDestroy()
{
	if (InternalWebSocket.IsValid())
	{
		InternalWebSocket->OnConnected().RemoveAll(this);
		InternalWebSocket->OnConnectionError().RemoveAll(this);
		InternalWebSocket->OnClosed().RemoveAll(this);
		InternalWebSocket->OnMessage().RemoveAll(this);
		InternalWebSocket->OnBinaryMessage().RemoveAll(this);
		InternalWebSocket->OnMessageSent().RemoveAll(this);
		InternalWebSocket.Reset();
	}

	Super::BeginDestroy();
}

void UWebSocket::OnWebSocketConnected_Internal()
{
	OnWebSocketConnected.Broadcast();
}

void UWebSocket::OnWebSocketConnectionError_Internal(const FString& Error)
{
	OnWebSocketConnectionError.Broadcast(Error);
}

void UWebSocket::OnWebSocketClosed_Internal(int32 StatusCode, const FString& Reason, bool bWasClean)
{
	OnWebSocketClosed.Broadcast(StatusCode, Reason, bWasClean);
}

void UWebSocket::OnWebSocketMessageReceived_Internal(const FString& Message)
{
	OnWebSocketMessageReceived.Broadcast(Message);
}

void UWebSocket::OnWebSocketBinaryMessageReceived_Internal(const void* Data, SIZE_T Size, bool bIsLastFragment)
{
	TArray<uint8> MessageData;
	CopySocketBytes(MessageData, Data, Size);
	OnWebSocketBinaryMessageReceived.Broadcast(MessageData, bIsLastFragment);
}

void UWebSocket::OnWebSocketMessageSent_Internal(const FString& Message)
{
	OnWebSocketMessageSent.Broadcast(Message);
}
