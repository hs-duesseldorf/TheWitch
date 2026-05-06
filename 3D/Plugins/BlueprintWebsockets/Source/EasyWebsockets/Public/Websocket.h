// Copyright 2022 Chris Ringenberg https://www.ringenberg.dev/

#pragma once

#include "CoreMinimal.h"
#include "UObject/Object.h"

#include "Websocket.generated.h"

class IWebSocket;

DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnWebSocketConnected);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnWebSocketConnectionError, const FString&, Error);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(FOnWebSocketClosed, int32, StatusCode, const FString&, Reason, bool, bWasClean);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnWebSocketMessageReceived, const FString&, Message);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnWebSocketBinaryMessageReceived, const TArray<uint8>&, Data, bool, bIsLastFragment);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnWebSocketMessageSent, const FString&, Message);

UCLASS(MinimalAPI, BlueprintType)
class UWebSocket final : public UObject
{
	GENERATED_BODY()

public:
	
	UPROPERTY(BlueprintAssignable, Category = "Easy WebSockets|Events")
	FOnWebSocketConnected OnWebSocketConnected;

	UPROPERTY(BlueprintAssignable, Category = "Easy WebSockets|Events")
	FOnWebSocketConnectionError OnWebSocketConnectionError;

	UPROPERTY(BlueprintAssignable, Category = "Easy WebSockets|Events")
	FOnWebSocketClosed OnWebSocketClosed;

	UPROPERTY(BlueprintAssignable, Category = "Easy WebSockets|Events")
	FOnWebSocketMessageReceived OnWebSocketMessageReceived;

	UPROPERTY(BlueprintAssignable, Category = "Easy WebSockets|Events")
	FOnWebSocketBinaryMessageReceived OnWebSocketBinaryMessageReceived;

	UPROPERTY(BlueprintAssignable, Category = "Easy WebSockets|Events")
	FOnWebSocketMessageSent OnWebSocketMessageSent;

	void InitWebSocket(TSharedPtr<IWebSocket> InWebSocket);

	UFUNCTION(BlueprintCallable, Category = "Easy WebSockets|Connection")
	void Connect();

	UFUNCTION(BlueprintCallable, Category = "Easy WebSockets|Connection")
	void Close(int32 StatusCode = 1000, const FString& Reason = TEXT(""));

	UFUNCTION(BlueprintPure, Category = "Easy WebSockets|Connection")
	bool IsConnected() const;

	UFUNCTION(BlueprintCallable, Category = "Easy WebSockets|Send")
	void SendMessage(const FString& Message);

	UFUNCTION(BlueprintCallable, Category = "Easy WebSockets|Send")
	void SendBinaryMessage(const TArray<uint8>& Data);

private:

	virtual void BeginDestroy() override;

	UFUNCTION()
	void OnWebSocketConnected_Internal();

	UFUNCTION()
	void OnWebSocketConnectionError_Internal(const FString& Error);

	UFUNCTION()
	void OnWebSocketClosed_Internal(int32 StatusCode, const FString& Reason, bool bWasClean);

	UFUNCTION()
	void OnWebSocketMessageReceived_Internal(const FString& Message);

	void OnWebSocketBinaryMessageReceived_Internal(const void* Data, SIZE_T Size, bool bIsLastFragment);

	UFUNCTION()
	void OnWebSocketMessageSent_Internal(const FString& Message);
	
	TSharedPtr<IWebSocket> InternalWebSocket;
	
};
